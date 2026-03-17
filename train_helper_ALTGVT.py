import os
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.dataloader import default_collate
import numpy as np
from datetime import datetime
import torch.nn.functional as F
from datasets.crowd import Crowd_qnrf, Crowd_nwpu, Crowd_sh, CustomDataset, Crowd_jhu
from timm.scheduler.cosine_lr import CosineLRScheduler
from utils.early_stopping import EarlyStopping
#from Networks import vgg19
#from Networks import vgg_graph
#from Networks import ALTGVT
from Networks import mpvit
#from Networks import mpvit_vheat
from losses.ot_loss import OT_Loss
from losses.bay_loss import Bay_Loss
from losses.post_prob import Post_Prob

from utils.pytorch_utils import Save_Handle, AverageMeter
import utils.log_utils as log_utils
import wandb

from math import ceil

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import ptflops


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[
        1
    ]  # the number of points is not fixed, keep it as a list of tensor
    gt_discretes = torch.stack(transposed_batch[2], 0)
    return images, points, gt_discretes

# def train_collate(batch):    #改gt
#     transposed_batch = list(zip(*batch))
#     images = torch.stack(transposed_batch[0], 0)
#     points = transposed_batch[
#         1
#     ]  # the number of points is not fixed, keep it as a list of tensor
#     targets = transposed_batch[2]
#     st_size = torch.FloatTensor(transposed_batch[3])
#     return images, points, targets, st_size

class Trainer(object):
    def __init__(self, args):
        self.args = args

    def setup(self): #其中的 setup 方法用于设置训练的相关参数和目录结构
        args = self.args
        #---------改为并行-----------------------------------------------------------------------
        # dist.init_process_group(backend="nccl")
        # local_rank = int(os.environ["LOCAL_RANK"])
        # torch.cuda.set_device(local_rank)
        # self.device = torch.device("cuda", local_rank)

        # self.model = mpvit_vheat.mpvit_small(pretrained=True).to(self.device)
        # # wrap
        # self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank)
        #-----------------------------------------------------------------------------------
        sub_dir = (
            "mpvit-Oct/{}_12-1-input-{}_wot-{}_wtv-{}_reg-{}_nIter-{}_normCood-{}".format(
                args.run_name,
                args.crop_size,
                args.wot,
                args.wtv,

                args.reg,
                args.num_of_iter_in_ot,
                args.norm_cood,
            )
        )

        self.save_dir = os.path.join("ckpts", sub_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        time_str = datetime.strftime(datetime.now(), "%m%d-%H%M%S")
        self.logger = log_utils.get_logger(
            os.path.join(self.save_dir, "train-{:s}.log".format(time_str))
        )
        log_utils.print_config(vars(args), self.logger)

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.device_count = torch.cuda.device_count()
            #assert self.device_count == 10
            self.logger.info("using {} gpus".format(self.device_count))
        # if torch.cuda.is_available():
        #     # 用所有 GPU
        #     self.device_count = torch.cuda.device_count()
        #     self.logger.info(f"found {self.device_count} gpus, using DataParallel")
        #     self.device = torch.device("cuda")  # 主卡默认为 cuda:0
        else:
            raise Exception("gpu is not available")

        downsample_ratio = 8
        if args.dataset.lower() == "qnrf":
            self.datasets = {
                x: Crowd_qnrf(
                    os.path.join(
                        args.data_dir, x), args.crop_size, downsample_ratio, x
                )
                for x in ["train", "val"]
            }
        elif args.dataset.lower() == "nwpu":
            self.datasets = {
                x: Crowd_nwpu(
                    os.path.join(
                        args.data_dir, x), args.crop_size, downsample_ratio, x
                )
                for x in ["train", "val"]
            }
        elif args.dataset.lower() == "sha" or args.dataset.lower() == "shb":
            self.datasets = {
                "train": Crowd_sh(
                    os.path.join(args.data_dir, "train_data"),
                    args.crop_size,
                    downsample_ratio,
                    # False,
                    "train",
                ),
                "val": Crowd_sh(
                    os.path.join(args.data_dir, "test_data"),
                    args.crop_size,
                    downsample_ratio,
                    # False,
                    "val",
                ),
            }
        elif args.dataset.lower() == "jhu":
            self.datasets = {
                x: Crowd_jhu(
                    os.path.join(
                        args.data_dir, x), args.crop_size, downsample_ratio, x
                )
                for x in ["train", "val"]
            }
        elif args.dataset.lower() == "custom":
            self.datasets = {
                "train": CustomDataset(
                    args.data_dir, args.crop_size, downsample_ratio, method="train"
                ),
                "val": CustomDataset(
                    args.data_dir, args.crop_size, downsample_ratio, method="valid"
                ),
            }
        else:
            raise NotImplementedError

        self.dataloaders = {
            x: DataLoader(
                self.datasets[x],
                collate_fn=(train_collate if x ==
                            "train" else default_collate),
                batch_size=(args.batch_size if x == "train" else 1),
                shuffle=(True if x == "train" else False),
                num_workers=args.num_workers * self.device_count,
                pin_memory=(True if x == "train" else False),
            )
            for x in ["train", "val"]
        }

        # 在train_helper_ALTGVT.py中分别设置
        # self.train_batch_size = 4  # 训练batch size
        # self.val_batch_size = 2  # 验证batch


        # train_ds = self.datasets["train"]
        # train_im_list = train_ds.im_list  #这是
        # img_counts = []
        # for p in train_im_list:
        #     gd_path = p.replace('jpg', 'npy')
        #     pts = np.load(gd_path)
        #     img_counts.append(len(pts))
        # img_counts = np.array(img_counts, dtype=np.float32)
        # img_counts = img_counts + 1e-6
        # weight = img_counts / img_counts.sum()
        # sample_weights = torch.from_numpy(weight).float()
        # sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_ds), replacement=True)
        #
        # train_loader = DataLoader(
        #     train_ds,
        #     batch_size=args.batch_size,
        #     sampler=sampler,
        #     collate_fn=train_collate,
        #     num_workers=max(1, args.num_workers * self.device_count),
        #     pin_memory=True,
        # )
        #
        # def val_collate(batch):
        #     assert len(batch) == 1
        #     return batch[0]
        #
        # val_loader = DataLoader(
        #     self.datasets["val"],
        #     batch_size=1,
        #     shuffle= False,
        #     collate_fn=val_collate,
        #     num_workers=max(1, args.num_workers * self.device_count),
        #     pin_memory=False,
        # )
        # self.dataloaders = {"train": train_loader, "val": val_loader}

        self.model = mpvit.mpvit_small(pretrained=True)
        self.model.to(self.device)
        macs, params = ptflops.get_model_complexity_info(self.model, (3, self.args.crop_size, self.args.crop_size),
                                                         as_strings=True, print_per_layer_stat=True)
        self.logger.info(f'Computational complexity(macs): {macs}')
        self.logger.info(f'Number of parameters(params): {params}')
        #self.model = ALTGVT.alt_gvt_large(pretrained=True)
        #self.model = mpvit_vheat.mpvit_small(pretrained=True)
        #self.model = vgg_graph.vgg19_trans(topk=args.topk, usenum=args.usenum, pretrained=True)
        #self.model = vgg19.vgg19(pretrained=True)

        # self.model.load_state_dict(torch.load('/home/ln/A/JXX/crowd_counting/autodl-tmp/ckpts/mask_biformer/JHU-only-test_12-1-input-512_wot-0.1_wtv-0.01_reg-10.0_nIter-100_normCood-0/best_model.pth', self.device))

        #self.model = mpvit_vheat.mpvit_small(pretrained=True)

        # self.model = mpvit_vheat.mpvit_small(pretrained=True, img_size=args.crop_size)
        # # DataParallel 会把 batch 拆到多个卡上
        # self.model = nn.DataParallel(self.model)
        # self.model.to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        # 初始化学习率策略
        if args.lr_scheduler:
            self.logger.info('Initializing learning rate scheduler')
            n_iter_per_epoch = len(self.dataloaders['train'])
            self.n_iter_per_epoch = n_iter_per_epoch
            num_steps = int(args.max_epoch * n_iter_per_epoch)
            warmup_steps = int(args.warmup_epochs * n_iter_per_epoch)
            decay_steps = int(args.decay_epochs * n_iter_per_epoch)
            self.scheduler = CosineLRScheduler(
                self.optimizer,
                t_initial=num_steps,
                # t_mul=1.0,
                lr_min=args.lr_min,
                warmup_lr_init=args.warmup_lr,
                warmup_t=warmup_steps,
                cycle_limit=1,
                t_in_epochs=False,
            )
        self.start_epoch = 0

        # 早停类
        if args.early_stopping:
            self.logger.info('Initializing early stopping')
            self.early_stopping = EarlyStopping(patience=args.patient, delta=1, trace_func=self.logger.info)

        # 如果启用了 WandB（Weights and Biases）日志记录，初始化 WandB，并记录配置信息。如果没有启用 WandB，则禁用 WandB
        if args.wandb:
            wandb.login(key='7b19069b335990f6fa58cc576c72a49589cf7139')
            self.wandb_run = wandb.init(
                config=args, project="Crowd Counting", name=args.run_name
            )
        else:
            wandb.init(mode="disabled")

        # check if wandb has to log
        if args.wandb:
            self.wandb_run = wandb.init(
            config=args, project="CTTrans", name=args.run_name
        )
        else :
            wandb.init(mode="disabled")


        if args.resume:
            self.logger.info("loading pretrained model from " + args.resume)
            suf = args.resume.rsplit(".", 1)[-1]
            if suf == "tar":
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(
                    checkpoint["optimizer_state_dict"])
                self.start_epoch = checkpoint["epoch"] + 1
            elif suf == "pth":
                self.model.load_state_dict(
                    torch.load(args.resume, self.device))
        else:
            self.logger.info("random initialization")

        self.ot_loss = OT_Loss(
            args.crop_size,
            downsample_ratio,
            args.norm_cood,
            self.device,
            args.num_of_iter_in_ot,
            args.reg,
        )


        #self.tv_loss = nn.L1Loss(reduction="none").to(self.device)

        self.tv_loss = nn.MSELoss(reduction="none").to(self.device)

        #换生成gt方式新加的---------------------------------------------------------------
        # self.save_all = args.save_all
        # self.post_prob = Post_Prob(
        #     args.sigma,
        #     args.crop_size,
        #     args.downsample_ratio,
        #     args.background_ratio,
        #     args.use_background,
        #     self.device
        # )
        # self.criterion = Bay_Loss(args.use_background, self.device)
        #-----------------------------------------------------------------------------

        self.mse = nn.MSELoss().to(self.device)
        self.mae = nn.L1Loss().to(self.device)
        self.save_list = Save_Handle(max_num=1)
        self.best_mae = np.inf
        self.best_mse = np.inf
        # self.best_count = 0

    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch + 1):
            self.logger.info(
                "-" * 5 + "Epoch {}/{}".format(epoch, args.max_epoch) + "-" * 5
            )
            self.epoch = epoch
            self.train_epoch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()

    # train_epoch和val_epoch都换了，原本的被注释
    # def train_epoch(self):
    #     epoch_loss = AverageMeter()
    #     epoch_mae = AverageMeter()
    #     epoch_mse = AverageMeter()
    #     epoch_start = time.time()
    #     self.model.train()  # Set model to training mode
    #
    #     # Iterate over data.
    #     for step, (inputs, points, targets, st_sizes) in enumerate(self.dataloaders["train"]):
    #         inputs = inputs.to(self.device)
    #         st_sizes = st_sizes.to(self.device)
    #         gd_count = np.array([len(p) for p in points], dtype=np.float32)
    #         points = [p.to(self.device) for p in points]
    #         targets = [t.to(self.device) for t in targets]
    #
    #         with torch.set_grad_enabled(True):
    #             outputs, pe_list = self.model(inputs)
    #             prob_list = self.post_prob(points, st_sizes)
    #             loss = self.criterion(prob_list, targets, outputs)
    #             for pe in pe_list:
    #                 loss_pe = torch.var(pe, dim=-1)
    #                 # loss_pe = torch.sum(loss_pe[loss_pe>0.1])
    #                 loss_pe = torch.sum(loss_pe)
    #                 loss += 0.1 * loss_pe
    #             loss.backward()
    #
    #         if True:
    #             self.optimizer.step()
    #             self.optimizer.zero_grad()
    #
    #             N = inputs.size(0)
    #             pre_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
    #             res = pre_count - gd_count
    #             epoch_loss.update(loss.item(), N)
    #             epoch_mse.update(np.mean(res * res), N)
    #             epoch_mae.update(np.mean(abs(res)), N)
    #
    #     self.logger.info('Epoch {} Train, Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
    #                  .format(self.epoch, epoch_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
    #                          time.time() - epoch_start))
    #     model_state_dic = self.model.state_dict()
    #     save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
    #     torch.save({
    #         'epoch': self.epoch,
    #         'optimizer_state_dict': self.optimizer.state_dict(),
    #         'model_state_dict': model_state_dic
    #     }, save_path)
    #     self.save_list.append(save_path)  # control the number of saved models
    #
    # def val_epoch(self):
    #     #print(f"11Validation dataloader type: {type(self.dataloaders['val'])}")
    #     epoch_start = time.time()
    #     self.model.eval()  # Set model to evaluate mode
    #     epoch_res = []
    #
    #     # print(f"22Validation dataloader type: {type(self.dataloaders['val'])}")
    #     # for i, batch in enumerate(self.dataloaders["val"]):
    #     #     print(f"Batch {i} type: {type(batch)}")
    #     #     print(f"Batch {i} length: {len(batch)}")
    #     #
    #     #     # 临时：打印每个元素的类型和形状
    #     #     for j, item in enumerate(batch):
    #     #         if torch.is_tensor(item):
    #     #             print(f"  Item {j}: tensor with shape {item.shape}")
    #     #         else:
    #     #             print(f"  Item {j}: type {type(item)}")
    #
    #     # Iterate over data.
    #     #for inputs, count, name in self.dataloaders["val"]:
    #     for inputs, count, name in self.dataloaders["val"]:
    #         inputs = inputs.to(self.device)
    #         # st_sizes = st_sizes.to(self.device)
    #         # gd_count = np.array([len(p) for p in points], dtype=np.float32)
    #         # points = [p.to(self.device) for p in points]
    #         # targets = [t.to(self.device) for t in targets]
    #         # inputs are images with different sizes
    #         b, c, h, w = inputs.shape
    #         h, w = int(h), int(w)
    #         assert b == 1, 'the batch size should equal to 1 in validation mode'
    #         input_list = []
    #         c_size = 1024
    #         if h >= c_size or w >= c_size:
    #             h_stride = int(ceil(1.0 * h / c_size))
    #             w_stride = int(ceil(1.0 * w / c_size))
    #             h_step = h // h_stride
    #             w_step = w // w_stride
    #             for i in range(h_stride):
    #                 for j in range(w_stride):
    #                     h_start = i * h_step
    #                     if i != h_stride - 1:
    #                         h_end = (i + 1) * h_step
    #                     else:
    #                         h_end = h
    #                     w_start = j * w_step
    #                     if j != w_stride - 1:
    #                         w_end = (j + 1) * w_step
    #                     else:
    #                         w_end = w
    #                     input_list.append(inputs[:, :, h_start:h_end, w_start:w_end])
    #             with torch.set_grad_enabled(False):
    #                 pre_count = 0.0
    #                 for idx, input in enumerate(input_list):
    #                     output = self.model(input)[0]
    #                     pre_count += torch.sum(output)
    #             res = count[0].item() - pre_count.item()
    #             epoch_res.append(res)
    #         else:
    #             with torch.set_grad_enabled(False):
    #                 outputs = self.model(inputs)[0]
    #                 # save_results(inputs, outputs, self.vis_dir, '{}.jpg'.format(name[0]))
    #                 res = count[0].item() - torch.sum(outputs).item()
    #                 epoch_res.append(res)
    #
    #     epoch_res = np.array(epoch_res)
    #     mse = np.sqrt(np.mean(np.square(epoch_res)))
    #     mae = np.mean(np.abs(epoch_res))
    #     self.logger.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
    #                  .format(self.epoch, mse, mae, time.time() - epoch_start))
    #
    #     model_state_dic = self.model.state_dict()
    #     self.logger.info("best mse {:.2f} mae {:.2f}".format(self.best_mse, self.best_mae))
    #     if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
    #         self.best_mse = mse
    #         self.best_mae = mae
    #         self.logger.info(
    #             "save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse, self.best_mae, self.epoch))
    #         if self.save_all:
    #             torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.best_count)))
    #             self.best_count += 1
    #         else:
    #             torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_mse-{:.2f}-mae-{:.2f}_epoch-{}.pth'.format(self.best_mse, self.best_mae, self.epoch)))
    #
    #             #         model_path = os.path.join(
    #             #             self.save_dir, "best_model_mae-{:.2f}_epoch-{}.pth".format(
    #             #                 self.best_mae, self.epoch)
    #             #         )

    def train_epoch(self):
        epoch_ot_loss = AverageMeter()
        epoch_ot_obj_value = AverageMeter()
        epoch_wd = AverageMeter()
        epoch_count_loss = AverageMeter()
        epoch_tv_loss = AverageMeter()
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.train()  # Set model to training mode

        for step, (inputs, points, gt_discrete) in enumerate(self.dataloaders["train"]):
            inputs = inputs.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            gt_discrete = gt_discrete.to(self.device)
            N = inputs.size(0)

            with torch.set_grad_enabled(True):
                outputs, outputs_normed = self.model(inputs)

                # 添加调试信息
                # print(f"[Train] Input size: {inputs.size()}")
                # print(f"[Train] Model outputs size: {outputs.size()}")
                # print(f"[Train] Model outputs_normed size: {outputs_normed.size()}")
                # print(f"[Train] GT discrete size: {gt_discrete.size()}")

                # Compute OT loss.
                ot_loss, wd, ot_obj_value = self.ot_loss(
                    outputs_normed, outputs, points
                )
                ot_loss = ot_loss * self.args.wot
                ot_obj_value = ot_obj_value * self.args.wot
                epoch_ot_loss.update(ot_loss.item(), N)
                epoch_ot_obj_value.update(ot_obj_value.item(), N)
                epoch_wd.update(wd, N)

                # Compute counting loss.
                count_loss = self.mae(
                    outputs.sum(1).sum(1).sum(1),
                    torch.from_numpy(gd_count).float().to(self.device),
                )
                epoch_count_loss.update(count_loss.item(), N)


                #1.Compute TV loss.
                # gd_count_tensor = (
                #     torch.from_numpy(gd_count)
                #     .float()
                #     .to(self.device)
                #     .unsqueeze(1)
                #     .unsqueeze(2)
                #     .unsqueeze(3)
                # )
                # gt_discrete_normed = gt_discrete / (gd_count_tensor + 1e-6)
                # tv_loss = (
                #     self.tv_loss(outputs_normed, gt_discrete_normed)
                #     .sum(1)
                #     .sum(1)
                #     .sum(1)
                #     * torch.from_numpy(gd_count).float().to(self.device)
                # ).mean(0) * self.args.wtv
                # epoch_tv_loss.update(tv_loss.item(), N)

                #2.Compute TV loss（修改后）
                gd_count_tensor = (
                    torch.from_numpy(gd_count)
                    .float()
                    .to(self.device)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                )
                gt_discrete_normed = gt_discrete / (gd_count_tensor + 1e-6)


                #计算MSE损失并调整维度
                # tv_loss = (
                #     self.tv_loss(outputs_normed, gt_discrete_normed)  # 形状 [N, C, H, W]
                #     .sum(dim=[1,2,3])  # 合并CHW维度 → [N]
                #     * torch.from_numpy(gd_count).float().to(self.device)  # [N]
                # ).mean() * self.args.wtv  # 标量
                # epoch_tv_loss.update(tv_loss.item(), N)

                # 3.修改为混合损失
                tv_l1 = nn.L1Loss(reduction='none')(outputs_normed, gt_discrete_normed)  # 保持元素级损失
                tv_mse = nn.MSELoss(reduction='none')(outputs_normed, gt_discrete_normed)

                # 混合计算（建议比例：0.3 L1 + 0.7 MSE）
                mixed_loss = 0.3 * tv_l1 + 0.7 * tv_mse

                tv_loss = (
                    mixed_loss.sum(dim=[1,2,3])  # 沿CHW维度求和
                    * torch.from_numpy(gd_count).float().to(self.device)
                ).mean() * self.args.wtv  # 调整权重系数
                epoch_tv_loss.update(tv_loss.item(), N)

                loss = ot_loss + count_loss + tv_loss

                #loss = count_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pred_count = (
                    torch.sum(outputs.view(N, -1),
                              dim=1).detach().cpu().numpy()
                )
                pred_err = pred_count - gd_count
                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(pred_err * pred_err), N)
                epoch_mae.update(np.mean(abs(pred_err)), N)

                # log wandb
                # wandb.log(
                #     {
                #         "train/TOTAL_loss": loss,
                #         "train/count_loss": count_loss,
                #         "train/tv_loss": tv_loss,
                #         "train/pred_err": pred_err,
                #     },
                #     step=self.epoch,
                # )


                # 修改后（取消注释并增强）：
                # wandb.log(
                #     {
                #         "train/TOTAL_loss": loss.item(),
                #         "train/count_loss": count_loss.item(),
                #         "train/tv_loss": tv_loss.item(),
                #         "train/OT_loss": ot_loss.item(),
                #         "train/MAE": epoch_mae.get_avg(),
                #         "train/MSE": np.sqrt(epoch_mse.get_avg()),
                #         "lr": self.optimizer.param_groups[0]['lr']  # 可选添加学习率监控
                #     },
                #     step=step + self.epoch * len(self.dataloaders["train"])  # 改为全局step计数
                # )


        self.logger.info(
            "Epoch {} Train, Loss: {:.2f}, OT Loss: {:.2e}, Wass Distance: {:.2f}, OT obj value: {:.2f}, "
            "Count Loss: {:.2f}, TV Loss: {:.5f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec".format(
                self.epoch,
                epoch_loss.get_avg(),
                epoch_ot_loss.get_avg(),
                epoch_wd.get_avg(),
                epoch_ot_obj_value.get_avg(),
                epoch_count_loss.get_avg(),
                epoch_tv_loss.get_avg(),

                np.sqrt(epoch_mse.get_avg()),
                epoch_mae.get_avg(),
                time.time() - epoch_start,
            )
        )
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(
            self.save_dir, "{}_ckpt.tar".format(self.epoch))
        torch.save(
            {
                "epoch": self.epoch,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_state_dict": model_state_dic,
            },
            save_path,
        )
        self.save_list.append(save_path)

    # ---------- sliding-window 推理函数（保持和你的 downsample_ratio 一致） ----------
    # def sliding_window_inference(self, pil_img, trans, crop_size, downsample_ratio=8, stride=None):
    #     """
    #     model: your model, must accept tensor (1,C,h,w) and return density map (1,1,h_out,w_out)
    #     pil_img: PIL.Image (原始分辨率)
    #     trans: the same transforms.Compose used in your dataset (ToTensor + Normalize)
    #     crop_size: patch 大小（像素），应与训练时 self.c_size 一致或接近
    #     downsample_ratio: 模型输出 / 原始图像的下采样比（与你的 d_ratio 保持一致）
    #     stride: 滑窗步长（像素），如果 None 则默认 0.75*crop_size
    #     返回: pred_down_map (numpy array shape: [down_h, down_w]), pred_count (float)
    #     """
    #
    #     if stride is None:
    #         stride = int(crop_size * 0.75)
    #     if crop_size is None:
    #         crop_size = self.args.crop_size
    #     if trans is None:
    #         trans = self.datasets['val'].trans
    #
    #     device = self.device
    #     model =self.model
    #     model.eval()
    #
    #     W, H = pil_img.size  # PIL: width, height
    #     down_h = H // downsample_ratio
    #     down_w = W // downsample_ratio
    #
    #     score_map = torch.zeros((1, 1, down_h, down_w), dtype=torch.float32, device=device)
    #     weight_map = torch.zeros_like(score_map)
    #
    #     # 预先计算 full-size hann window（在 downsample 空间）
    #     # 对每个 patch，我们会把 pred 下采样到 patch_down_h x patch_down_w，然后用 hann 窗平滑融合
    #     hann_cache = {}  # 以 (ph_down, pw_down) 为 key 缓存 hann 窗
    #
    #     for y in range(0, H, stride):
    #         for x in range(0, W, stride):
    #             y0 = y
    #             x0 = x
    #             y1 = min(y0 + crop_size, H)
    #             x1 = min(x0 + crop_size, W)
    #             # 保证 patch 大小为 crop_size（边界处回退）
    #             y0 = max(0, y1 - crop_size)
    #             x0 = max(0, x1 - crop_size)
    #             patch = pil_img.crop((x0, y0, x1, y1))  # (left, upper, right, lower)
    #             patch_t = trans(patch).unsqueeze(0).to(device)
    #
    #             # 预处理并送入模型
    #             with torch.no_grad():
    #                 out = model(patch_t)
    #                 if isinstance(out, (tuple, list)):
    #                     pred = out[0]
    #                 else:
    #                     pred = out
    #                 if not torch.is_tensor(pred):
    #                     raise RuntimeError(f"Model output first element is not a tensor, got {type(pred)}")
    #                 # patch_t = trans(patch).unsqueeze(0).to(device)  # 1,C,h,w
    #                 # pred = model(patch_t)  # 1,1,ph_out,pw_out  (模型输出可能已经是 downsampled)
    #                 # ph_out, pw_out = pred.shape[-2], pred.shape[-1]
    #
    #                 # 我们需要把 pred 对齐到 patch 在 downsample 空间的尺寸：
    #                 patch_h = y1 - y0
    #                 patch_w = x1 - x0
    #                 ph_down = patch_h // downsample_ratio
    #                 pw_down = patch_w // downsample_ratio
    #
    #                 ph_out, pw_out = pred.shape[-2], pred.shape[-1]
    #                 # 若模型输出与 (ph_down, pw_down) 不同，插值到期望尺寸
    #                 if (ph_out != ph_down) or (pw_out != pw_down):
    #                     pred = F.interpolate(pred, size=(ph_down, pw_down), mode='bilinear', align_corners=False)
    #
    #                 # 计算目标在 downsample map 中的位置
    #                 y0_down = y0 // downsample_ratio
    #                 x0_down = x0 // downsample_ratio
    #                 y1_down = y0_down + ph_down
    #                 x1_down = x0_down + pw_down
    #
    #                 # hann 窗（下采样尺寸）
    #                 key = (ph_down, pw_down)
    #                 if key in hann_cache:
    #                     win2d = hann_cache[key]
    #                 else:
    #                     wy = torch.hann_window(ph_down, device=device)
    #                     wx = torch.hann_window(pw_down, device=device)
    #                     win2d = (wy.unsqueeze(1) @ wx.unsqueeze(0)).unsqueeze(0).unsqueeze(0)  # 1,1,ph_down,pw_down
    #                     hann_cache[key] = win2d
    #
    #                 score_map[..., y0_down:y1_down, x0_down:x1_down] += pred * win2d
    #                 weight_map[..., y0_down:y1_down, x0_down:x1_down] += win2d
    #
    #     # 防止除零
    #     final = score_map / (weight_map + 1e-6)
    #     pred_down_map = final.squeeze(0).squeeze(0).cpu().numpy()  # [down_h, down_w]
    #     pred_count = float(pred_down_map.sum())
    #     return pred_down_map, float(pred_count)
    #

    def val_epoch(self):
        args = self.args
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_res = []
        for inputs, count, name in self.dataloaders["val"]:
            with torch.no_grad():
                # inputs = cal_new_tensor(inputs, min_size=args.crop_size)
                inputs = inputs.to(self.device)
                crop_imgs, crop_masks = [], []
                b, c, h, w = inputs.size()

                rh, rw = args.crop_size, args.crop_size
                for i in range(0, h, rh):
                    gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                    for j in range(0, w, rw):
                        gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                        crop_imgs.append(inputs[:, :, gis:gie, gjs:gje])
                        mask = torch.zeros([b, 1, h, w]).to(self.device)
                        mask[:, :, gis:gie, gjs:gje].fill_(1.0)
                        crop_masks.append(mask)
                crop_imgs, crop_masks = map(
                    lambda x: torch.cat(x, dim=0), (crop_imgs, crop_masks)
                )

                crop_preds = []
                # batch_size = self.val_batch_size  #由于内存溢出新加的
                nz, bz = crop_imgs.size(0), args.batch_size
                for i in range(0, nz, bz):
                    gs, gt = i, min(nz, i + bz)
                    crop_pred, _ = self.model(crop_imgs[gs:gt])

                    _, _, h1, w1 = crop_pred.size()
                    crop_pred = (
                        F.interpolate(
                            crop_pred,
                            size=(h1 * 8, w1 * 8),
                            mode="bilinear",
                            align_corners=True,
                        )
                        / 64
                    )

                    crop_preds.append(crop_pred)
                crop_preds = torch.cat(crop_preds, dim=0)

                # splice them to the original size
                idx = 0
                pred_map = torch.zeros([b, 1, h, w]).to(self.device)
                for i in range(0, h, rh):
                    gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                    for j in range(0, w, rw):
                        gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                        pred_map[:, :, gis:gie, gjs:gje] += crop_preds[idx]
                        idx += 1
                # for the overlapping area, compute average value
                mask = crop_masks.sum(dim=0).unsqueeze(0)
                outputs = pred_map / mask

                res = count[0].item() - torch.sum(outputs).item()
                epoch_res.append(res)
        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))

        self.logger.info(
            "Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec".format(
                self.epoch, mse, mae, time.time() - epoch_start
            )
        )

        # log wandb
        wandb.log({"val/MSE": mse, "val/MAE": mae}, step=self.epoch)

        model_state_dic = self.model.state_dict()
        # if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
        print("Comaprison", mae,  self.best_mae)
        if mae < self.best_mae:
            self.best_mse = mse
            self.best_mae = mae
            self.logger.info(
                "save best mse {:.2f} mae {:.2f} model epoch {}".format(
                    self.best_mse, self.best_mae, self.epoch
                )
            )
            print("Saving best model at {} epoch".format(self.epoch))
            model_path = os.path.join(
                self.save_dir, "best_model_mae-{:.2f}_epoch-{}.pth".format(
                    self.best_mae, self.epoch)
            )
            torch.save(
                model_state_dic,
                model_path,
            )

            if args.wandb:
                artifact = wandb.Artifact("model", type="model")
                artifact.add_file(model_path)

                self.wandb_run.log_artifact(artifact)

            # torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.best_count)))
            # self.best_count += 1


def tensor_divideByfactor(img_tensor, factor=32):
    _, _, h, w = img_tensor.size()
    h, w = int(h // factor * factor), int(w // factor * factor)
    img_tensor = F.interpolate(
        img_tensor, (h, w), mode="bilinear", align_corners=True)

    return img_tensor

def cal_new_tensor(img_tensor, min_size=256):   #应该取决于不同的数据集
    _, _, h, w = img_tensor.size()
    if min(h, w) < min_size:
        ratio_h, ratio_w = min_size / h, min_size / w
        if ratio_h >= ratio_w:
            img_tensor = F.interpolate(
                img_tensor,
                (min_size, int(min_size / h * w)),
                mode="bilinear",
                align_corners=True,
            )
        else:
            img_tensor = F.interpolate(
                img_tensor,
                (int(min_size / w * h), min_size),
                mode="bilinear",
                align_corners=True,
            )
    return img_tensor

#     def val_epoch(self):
#         args = self.args
#         epoch_start = time.time()
#         self.model.eval()
#         device = self.device
#
#         # Determine downsampling ratio (d_ratio) for validation dataset
#         val_dataset = self.datasets['val']
#         d_ratio = val_dataset.d_ratio if hasattr(val_dataset, 'd_ratio') else 8
#         crop_size = args.crop_size
#         stride = int(crop_size * 0.75)  # 75% overlap
#
#         residuals = []
#         for pil_img, keypoints, name in self.dataloaders["val"]:
#             gt_count = len(keypoints)
#
#             with torch.no_grad():
#                 # Process image using sliding window approach
#                 pred_down_map, pred_count = self.sliding_window_inference(pil_img, crop_size = self.args.crop_size, trans = self.datasets['val'].trans)
#
#             # Calculate residual for this image
#             residuals.append(gt_count - pred_count)
#
#         # Calculate epoch metrics
#         residuals = np.array(residuals)
#         mse = np.sqrt(np.mean(np.square(residuals)))
#         mae = np.mean(np.abs(residuals))
#         epoch_time = time.time() - epoch_start
#
#         # Log results
#         self.logger.info(
#             f"Epoch {self.epoch} Val, MSE: {mse:.2f} MAE: {mae:.2f}, Cost {epoch_time:.1f} sec"
#         )
#         wandb.log({"val/MSE": mse, "val/MAE": mae}, step=self.epoch)
#
#         # Save best model checkpoint
#         if mae < self.best_mae:
#             self.best_mse = mse
#             self.best_mae = mae
#             self.logger.info(
#                 f"save best mse {mse:.2f} mae {mae:.2f} model epoch {self.epoch}"
#             )
#
#             model_path = os.path.join(
#                 self.save_dir,
#                 f"best_model_mae-{mae:.2f}_epoch-{self.epoch}.pth"
#             )
#             torch.save(self.model.state_dict(), model_path)
#
#             if args.wandb:
#                 artifact = wandb.Artifact("model", type="model")
#                 artifact.add_file(model_path)
#                 self.wandb_run.log_artifact(artifact)
#
#
# def _process_image(self, inputs, crop_size, stride, d_ratio):
#     """Process single image using sliding window approach with Hann window weighting"""
#     _, _, H, W = inputs.size()
#     down_h = H // d_ratio
#     down_w = W // d_ratio
#
#     # Initialize accumulators for score and weight maps
#     score_map = torch.zeros((1, 1, down_h, down_w), device=inputs.device)
#     weight_map = torch.zeros_like(score_map)
#     hann_cache = {}  # Cache for Hann windows
#
#     # Process each patch in sliding window
#     for y in range(0, H, stride):
#         y0 = max(min(H - crop_size, y), 0)
#         y1 = min(H, y + crop_size)
#         for x in range(0, W, stride):
#             x0 = max(min(W - crop_size, x), 0)
#             x1 = min(W, x + crop_size)
#
#             # Extract patch and calculate downsampled dimensions
#             patch = inputs[:, :, y0:y1, x0:x1]
#             ph, pw = y1 - y0, x1 - x0
#             ph_down, pw_down = ph // d_ratio, pw // d_ratio
#
#             if ph_down <= 0 or pw_down <= 0:
#                 continue
#
#             # Model prediction and interpolation
#             pred_patch, _ = self.model(patch)
#             pred_patch = F.interpolate(
#                 pred_patch,
#                 size=(ph_down, pw_down),
#                 mode='bilinear',
#                 align_corners=False
#             )
#
#             # Get or create Hann window
#             win2d = hann_cache.get((ph_down, pw_down))
#             if win2d is None:
#                 wy = torch.hann_window(ph_down, device=inputs.device)
#                 wx = torch.hann_window(pw_down, device=inputs.device)
#                 win2d = wy.unsqueeze(1) @ wx.unsqueeze(0)
#                 win2d = win2d.unsqueeze(0).unsqueeze(0)
#                 hann_cache[(ph_down, pw_down)] = win2d
#
#             # Update score and weight maps
#             y0_d, x0_d = y0 // d_ratio, x0 // d_ratio
#             y1_d, x1_d = y0_d + ph_down, x0_d + pw_down
#
#             score_map[..., y0_d:y1_d, x0_d:x1_d] += pred_patch * win2d
#             weight_map[..., y0_d:y1_d, x0_d:x1_d] += win2d
#
#     # Combine results and return predicted count
#     density_map = score_map / (weight_map + 1e-6)
#     return density_map.sum().item()

if __name__ == "__main__":
    import torch

    print(torch.__file__)
    x = torch.ones(1, 3, 768, 1152)
    y = torch.tensor_split(x)
    print(y.size())
