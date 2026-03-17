import argparse
import os
import torch
from train_helper_ALTGVT import Trainer

# cd CCTrans-main 
# conda activate jxx
# python train.py --batch-size 4 --run-name jhu_mpvit+EMA+mask+coord_att+注意力融合 --max-epoch 2000 --val-epoch 2 --val-start 0
#/home/ln/A/JXX/cluster/data/QNRF-Train-Test
#/home/ln/A/JXX/crowd_counting/data/jhu_crowd_v2.0_1
#/home/ln/A/CJQ/CCTrans-main/data/part_A_final

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--data-dir', default='/home/ln/A/JXX/crowd_counting/data/jhu_crowd_v2.0_1', help='data path')
    parser.add_argument('--dataset', default='jhu', help='dataset name: qnrf, nwpu, sha, shb, custom,jhu')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='the initial learning rate')
    parser.add_argument('--lr_scheduler', type=int, default=0,
                        help='the initial lr scheduler')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='the weight decay')
    parser.add_argument('--resume', default='', type=str,
                        help='the path of resume training model')
    parser.add_argument('--max-epoch', type=int, default=2000, #sha 1000就可以了
                        help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=2, #每几轮验证一次，太大容易过拟合，训练结果好验证结果差
                        help='the num of steps to log training information')
    parser.add_argument('--val-start', type=int, default=0, #从第几轮开始验证，越大训练的越快，根据经验设置，sha50或100
                        help='the epoch start to val')
    parser.add_argument('--batch-size', type=int, default=1, #调试的时候设置成1 真正训练的时候1、2、4、8————，sha 8就可以了；shb设置的4 其他跑不了
                        help='train batch size')
    # 在train.py或配置文件中
    # parser.add_argument('--train_batch-size', type=int, default=4, help='training batch size')
    # parser.add_argument('--val_batch-size', type=int, default=2, help='validation batch size')

    parser.add_argument('--device', default='0,1,2,3,4,5,6,7,8,9', help='assign device')
    parser.add_argument('--num-workers', type=int, default=16,
                        help='the num of training process')
    parser.add_argument('--crop-size', type=int, default= 256,
                        help='the crop size of the train image')
    parser.add_argument('--wot', type=float, default=0.1, help='weight on OT loss')
    parser.add_argument('--wtv', type=float, default=0.001, help='weight on TV loss')
    parser.add_argument('--reg', type=float, default=10.0,
                        help='entropy regularization in sinkhorn')
    parser.add_argument('--num-of-iter-in-ot', type=int, default=100,
                        help='sinkhorn iterations')
    parser.add_argument('--norm-cood', type=int, default=0, help='whether to norm cood when computing distance')

    parser.add_argument('--run-name', default='CCTrans', help='run name for wandb interface/logging')
    parser.add_argument('--wandb', default=0, type=int, help='boolean to set wandb logging')

    parser.add_argument('--downsample-ratio', type=int, default=8,
                        help='downsample ratio')
    parser.add_argument('--use-background', type=bool, default=True,
                        help='whether to use background modelling')
    parser.add_argument('--sigma', type=float, default=8.0,
                        help='sigma for likelihood')
    parser.add_argument('--background-ratio', type=float, default=1.0,
                        help='background ratio')


    parser.add_argument('--warmup_epoch', type=int, default=10,
                        help='the warmup-epoch')
    parser.add_argument('--warmup_lr', type=int, default=5.0e-6,
                        help='the warmup_lr')
    parser.add_argument('--early_stopping', type=bool, default=False,
                        help='the initial earlystopping')

    # parser.add_argument('--save-all', type=bool, default=False,
    #                     help='whether to save all best model')  #林会


#下面两行是vgg_graph的参数
    # parser.add_argument('--topk', type=float, default=0.3, help='topk')
    # parser.add_argument('--usenum', type=int, default=18, help='usenum')   

    args = parser.parse_args()


    if args.dataset.lower() == 'qnrf':
        args.crop_size = 512
    elif args.dataset.lower() == 'nwpu':
        args.crop_size = 512 #这个数据集是512
        args.val_epoch = 2
    elif args.dataset.lower() == 'sha':
        args.crop_size = 256
    elif args.dataset.lower() == 'shb':
        args.crop_size = 512
    elif args.dataset.lower() == 'custom':
        args.crop_size = 256
    elif args.dataset.lower() == 'jhu':
        args.crop_size = 512
    else:
        raise NotImplementedError
    return args


if __name__ == '__main__':
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    trainer = Trainer(args)
    trainer.setup()
    trainer.train()
