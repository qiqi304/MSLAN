from PIL import Image
import torch.utils.data as data
import os
from glob import glob
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import numpy as np
import scipy.io as sio


def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w

def gen_discrete_map(im_height, im_width, points):
    """
        func: generate the discrete map.
        points: [num_gt, 2], for each row: [width, height]
        """
    discrete_map = np.zeros([im_height, im_width], dtype=np.float32)
    h, w = discrete_map.shape[:2]
    num_gt = points.shape[0]
    if num_gt == 0:
        return discrete_map
    
    # fast create discrete map
    points_np = np.array(points).round().astype(int)
    p_h = np.minimum(points_np[:, 1], np.array([h-1]*num_gt).astype(int))
    p_w = np.minimum(points_np[:, 0], np.array([w-1]*num_gt).astype(int))
    p_index = torch.from_numpy(p_h* im_width + p_w).to(torch.int64)
    discrete_map = torch.zeros(im_width * im_height).scatter_add_(0, index=p_index, src=torch.ones(im_width*im_height)).view(im_height, im_width).numpy()

    ''' slow method
    for p in points:
        p = np.round(p).astype(int)
        p[0], p[1] = min(h - 1, p[1]), min(w - 1, p[0])
        discrete_map[p[0], p[1]] += 1
    '''
    assert np.sum(discrete_map) == num_gt
    return discrete_map


class Base(data.Dataset):
    def __init__(self, root_path, crop_size, downsample_ratio=8):

        self.root_path = root_path
        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    def train_transform(self, img, keypoints):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        assert st_size >= self.c_size
        assert len(keypoints) >= 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        if len(keypoints) > 0:
            keypoints = keypoints - [j, i]
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                       (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])

        gt_discrete = gen_discrete_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        assert np.sum(gt_discrete) == len(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
                keypoints[:, 0] = w - keypoints[:, 0]
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
        gt_discrete = np.expand_dims(gt_discrete, 0)

        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(
            gt_discrete.copy()).float()


class Crowd_qnrf(Base):
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')))
        print('number of img: {}'.format(len(self.im_list)))
        if method not in ['train', 'val']:
            raise Exception("not implement")

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        gd_path = img_path.replace('jpg', 'npy')
        img = Image.open(img_path).convert('RGB')
        if self.method == 'train':
            keypoints = np.load(gd_path)
            return self.train_transform(img, keypoints)
        # elif self.method == 'val':
        #     keypoints = np.load(gd_path)
        #     name = os.path.basename(img_path).split('.')[0]
        #     return img, keypoints, name
        elif self.method == 'val':
            keypoints = np.load(gd_path)
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            return img, len(keypoints), name


class Crowd_jhu(Base):
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        self.im_list = sorted(glob(os.path.join(self.root_path, 'images', '*.jpg')))
        print('number of img: {}'.format(len(self.im_list)))
        if method not in ['train', 'val']:
            raise Exception("not implement")

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        gd_path = img_path.replace('jpg', 'npy')
        img = Image.open(img_path).convert('RGB')
        if self.method == 'train':
            keypoints = np.load(gd_path)
            return self.train_transform(img, keypoints)
        elif self.method == 'val':
            keypoints = np.load(gd_path)
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            return img, len(keypoints), name


class Crowd_nwpu(Base):
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')))
        print('number of img: {}'.format(len(self.im_list)))

        if method not in ['train', 'val', 'test']:
            raise Exception("not implement")

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        gd_path = img_path.replace('jpg', 'npy')
        img = Image.open(img_path).convert('RGB')
        if self.method == 'train':
            keypoints = np.load(gd_path)
            return self.train_transform(img, keypoints)
        elif self.method == 'val':
            keypoints = np.load(gd_path)
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            return img, len(keypoints), name
        elif self.method == 'test':
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            return img, name


class Crowd_sh(Base):
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        if method not in ['train', 'val']:
            raise Exception("not implement")

        self.im_list = sorted(glob(os.path.join(self.root_path, 'images', '*.jpg')))

        #print('number of img [{}]: {}'.format(method, len(self.im_list)))

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        name = os.path.basename(img_path).split('.')[0]
        gd_path = os.path.join(self.root_path, 'ground_truth', 'GT_{}.mat'.format(name))
        img = Image.open(img_path).convert('RGB')
        keypoints = sio.loadmat(gd_path)['image_info'][0][0][0][0][0]
        if self.method == 'train':
            return self.train_transform(img, keypoints)
        elif self.method == 'val':
            wd, ht = img.size
            st_size = 1.0 * min(wd, ht)
            if st_size < self.c_size:
                rr = 1.0 * self.c_size / st_size
                wd = round(wd * rr)
                ht = round(ht * rr)
                st_size = 1.0 * min(wd, ht)
                img = img.resize((wd, ht), Image.BICUBIC)
            img = self.trans(img)
            return img, len(keypoints), name
    # def __getitem__(self, item):
    #     img_path = self.im_list[item]
    #     gd_path = img_path.replace('jpg', 'npy')
    #     img = Image.open(img_path).convert('RGB')
    #     if self.method == 'train':
    #         keypoints = np.load(gd_path)
    #         return self.train_transform(img, keypoints)
    #     elif self.method == 'val':
    #         keypoints = np.load(gd_path)
    #         name = os.path.basename(img_path).split('.')[0]
    #         return img, keypoints,

    # def __getitem__(self, item):
    #     img_path = self.im_list[item]
    #     name = os.path.basename(img_path).split('.')[0]
    #     gd_path = os.path.join(self.root_path, 'ground_truth', 'GT_{}.mat'.format(name))
    #
    #     # 添加调试信息
    #     # print(f"[Dataset] Image path: {img_path}")
    #     # print(f"[Dataset] Ground truth path: {gd_path}")
    #
    #     img = Image.open(img_path).convert('RGB')
    #     keypoints = sio.loadmat(gd_path)['image_info'][0][0][0][0][0]
    #
    #     # print(f"[Dataset] Image size: {img.size}")
    #     # print(f"[Dataset] Number of keypoints: {len(keypoints)}")
    #
    #     if self.method == 'train':
    #         return self.train_transform(img, keypoints)
    #     elif self.method == 'val':
    #         wd, ht = img.size
    #         st_size = 1.0 * min(wd, ht)
    #         if st_size < self.c_size:
    #             rr = 1.0 * self.c_size / st_size
    #             wd = round(wd * rr)
    #             ht = round(ht * rr)
    #             st_size = 1.0 * min(wd, ht)
    #             img = img.resize((wd, ht), Image.BICUBIC)
    #         img = self.trans(img)
    #         return img, len(keypoints), name

    def train_transform(self, img, keypoints):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        # resize the image to fit the crop size
        if st_size < self.c_size:
            rr = 1.0 * self.c_size / st_size
            wd = round(wd * rr)
            ht = round(ht * rr)
            st_size = 1.0 * min(wd, ht)
            img = img.resize((wd, ht), Image.BICUBIC)
            keypoints = keypoints * rr
        assert st_size >= self.c_size, print(wd, ht)
        assert len(keypoints) >= 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        if len(keypoints) > 0:
            keypoints = keypoints - [j, i]
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                       (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])

        gt_discrete = gen_discrete_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        assert np.sum(gt_discrete) == len(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
                keypoints[:, 0] = w - keypoints[:, 0] - 1
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
        gt_discrete = np.expand_dims(gt_discrete, 0)

        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(
            gt_discrete.copy()).float()


# def cal_innner_area(c_left, c_up, c_right, c_down, bbox):  #计算裁剪区域与标注边界框的交集面积
#     #(c_left，c_up，c_right，c_down):裁剪区域的坐标; bbox: 人群边界框数据[N，4]，每行格式为[x1，y1，x2，y2]
#     inner_left = np.maximum(c_left, bbox[:, 0])
#     inner_up = np.maximum(c_up, bbox[:, 1])
#     inner_right = np.minimum(c_right, bbox[:, 2])
#     inner_down = np.minimum(c_down, bbox[:, 3])
#     inner_area = np.maximum(inner_right-inner_left, 0.0) * np.maximum(inner_down-inner_up, 0.0)
#     return inner_area
#
# class Crowd_sh(Base):
#     def __init__(self, root_path, crop_size,
#                  downsample_ratio=8, is_gray=False,
#                  method='train'):
#         super().__init__(root_path, crop_size, downsample_ratio)
#         self.method = method
#         if method not in ['train', 'val']:
#             raise Exception("not implement")
#
#         self.im_list = sorted(glob(os.path.join(self.root_path, 'images', '*.jpg')))
#
#         print('number of img [{}]: {}'.format(method, len(self.im_list)))
#
#         if is_gray:
#             self.trans = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#             ])
#         else:
#             self.trans = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#             ])
#
#     def __len__(self):
#         return len(self.im_list)
#
#     def __getitem__(self, item):
#         img_path = self.im_list[item]
#         name = os.path.basename(img_path).split('.')[0]
#         gd_path = os.path.join(self.root_path, 'ground_truth', 'GT_{}.mat'.format(name))
#         img = Image.open(img_path).convert('RGB')
#         keypoints = sio.loadmat(gd_path)['image_info'][0][0][0][0][0]
#         #print("Loaded keypoints shape:", keypoints.shape)  # 添加检查
#
#         if self.method == 'train':
#             return self.train_transform(img, keypoints)
#         elif self.method == 'val':
#             wd, ht = img.size
#             st_size = 1.0 * min(wd, ht)
#             if st_size < self.c_size:
#                 rr = 1.0 * self.c_size / st_size
#                 wd = round(wd * rr)
#                 ht = round(ht * rr)
#                 st_size = 1.0 * min(wd, ht)
#                 img = img.resize((wd, ht), Image.BICUBIC)
#             img = self.trans(img)
#             # print(f"Validation sample: "
#             #       f"img_shape={img.shape if isinstance(img, torch.Tensor) else img.size}, "
#             #       f"count={len(keypoints)}, name={name}")
#             return img, len(keypoints), name
#
#     def train_transform(self, img, keypoints): #数据增强
#         """random crop image patch and find people in it"""
#         wd, ht = img.size
#         # assert len(keypoints) > 0
#
#         # 1.随机灰度化（12%概率）
#         if random.random() > 0.88:
#             img = img.convert('L').convert('RGB')
#
#         #2.随机缩放 0.75~1.25
#         re_size = random.random() * 0.5 + 0.75
#         wdd = int(wd*re_size)
#         htt = int(ht*re_size)
#         keypoints = keypoints*re_size
#
#         #3.确保缩放后尺寸不小于c_size
#         if wdd < self.c_size:
#             htt = int(htt * self.c_size / wdd)
#             keypoints = keypoints*self.c_size / wdd
#             wdd = self.c_size
#         if htt < self.c_size:
#             wdd = int(wdd * self.c_size / htt)
#             keypoints = keypoints*self.c_size / htt
#             htt = self.c_size
#         wd = wdd
#         ht = htt
#         img = img.resize((wd, ht))
#         st_size = min(wd, ht)
#         assert st_size >= self.c_size
#
#         #4.随机裁剪
#         i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
#         img = F.crop(img, i, j, h, w)
#
#         # 5.裁剪区域中的人头可见性过滤
#         if len(keypoints) > 0:                                # 根据人头中心点和最近邻距离生成虚拟边界框
#             if len(keypoints) < 1000:
#                 # ====== 动态计算每个点的半径 (代替原始标注的第三列) ======
#                 # 计算点与点之间的距离矩阵
#                 dist_matrix = np.sqrt(np.sum((keypoints[:, None] - keypoints[None, :]) ** 2, axis=-1))
#
#                 # 将对角线设为无穷大，避免自己和自己比较
#                 np.fill_diagonal(dist_matrix, np.inf)
#
#                 # 获取每个点最近的邻居距离
#                 nearest_neighbor_dists = np.min(dist_matrix, axis=1)
#
#                 # 将最近邻距离的一半作为半径估计
#                 estimated_radius = nearest_neighbor_dists / 2.0
#
#                 # 限制半径在合理范围内 [4,40]
#                 nearest_dis = np.clip(estimated_radius, 4.0, 40.0)
#                 # ====== 动态计算结束 ======
#
#             else:  # 大规模点集使用KDTree加速
#                 from scipy.spatial import KDTree
#                 tree = KDTree(keypoints)
#                 nearest_neighbor_dists, _ = tree.query(keypoints, k=2)  # 每个点找最近的2个邻居
#                 nearest_dis = nearest_neighbor_dists[:, 1]  # 取第二个邻居（第一个是自己）
#
#             #nearest_dis = np.clip(keypoints[:, 2], 4.0, 40.0)             # 从标注获取人头半径限制在[4，40]
#             points_left_up = keypoints[:, :2] - nearest_dis[:, None] / 2.0             #人头边界左上角
#             points_right_down = keypoints[:, :2] + nearest_dis[:, None] / 2.0          # 右下角
#             bbox = np.concatenate((points_left_up, points_right_down), axis=1)  # 边界框
#
#         # 计算每个边界框在裁剪区域的可见比例
#             inner_area = cal_innner_area(j, i, j + w, i + h, bbox)  # 计算裁剪框与边界框交集面积
#             origin_area = nearest_dis * nearest_dis  # 原始人头区域面积
#             ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)  # 可见比例
#             mask = (ratio >= 0.5)  # 只保留可见比例大于50 % 的人头
#
#             target = ratio[mask]
#             keypoints = keypoints[mask]
#             keypoints = keypoints[:, :2] - [j, i]  # 坐标转换为裁剪后的局部坐标
#
#         # 6.随机水平翻转（50%概率）
#         if len(keypoints) > 0:
#             if random.random() > 0.5:
#                 img = F.hflip(img)
#                 keypoints[:, 0] = w - keypoints[:, 0]  # 同步翻转X坐标
#         else:
#             target = np.array([])
#             if random.random() > 0.5:
#                 img = F.hflip(img)
#         return self.trans(img), torch.from_numpy(keypoints.copy()).float(), \
#             torch.from_numpy(target.copy()).float(), st_size
#         # 返回：预处理图像img，人头坐标keypoints[N,2]，人头可见比例target_ratio[N]，原始短边尺寸
#
#         # 直接学习每个人头的精确位置，通过target_ratio加权处理遮挡情况，最终计数 = 预测点数


class CustomDataset(Base):
    '''
    Class that allows training for a custom dataset. The folder are designed in the following way:
    root_dataset_path:
        -> images_1
        ->another_folder_with_image
        ->train.list
        ->valid.list

    The content of the lists file (csv with space as separator) are:
        img_xx__path label_xx_path
        img_xx1__path label_xx1_path

    where label_xx_path contains a list of x,y position of the head.
    '''
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        if method not in ['train', 'valid', 'test']:
            raise Exception("not implement")

        # read the list file
        self.img_to_label = {}
        list_file = f'{method}.list' # train.list, valid.list or test.list
        with open(os.path.join(self.root_path, list_file)) as fin:
                for line in fin:
                    if len(line) < 2: 
                        continue
                    line = line.strip().split()
                    self.img_to_label[os.path.join(self.root_path, line[0].strip())] = \
                                    os.path.join(self.root_path, line[1].strip())
        self.img_list = sorted(list(self.img_to_label.keys()))


        print('number of img [{}]: {}'.format(method, len(self.img_list)))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img_path = self.img_list[item]
        gt_path = self.img_to_label[img_path]
        img_name = os.path.basename(img_path).split('.')[0]

        img = Image.open(img_path).convert('RGB')
        keypoints = self.load_head_annotation(gt_path)
       
        if self.method == 'train':
            return self.train_transform(img, keypoints)
        elif self.method == 'valid' or self.method == 'test':
            wd, ht = img.size
            st_size = 1.0 * min(wd, ht)
            if st_size < self.c_size:
                rr = 1.0 * self.c_size / st_size
                wd = round(wd * rr)
                ht = round(ht * rr)
                st_size = 1.0 * min(wd, ht)
                img = img.resize((wd, ht), Image.BICUBIC)
            img = self.trans(img)
            return img, len(keypoints), img_name

    def load_head_annotation(self, gt_path):
        annotations = []
        with open(gt_path) as annotation:
            for line in annotation:
                x = float(line.strip().split(' ')[0])
                y = float(line.strip().split(' ')[1])
                annotations.append([x, y])
        return np.array(annotations)

    def train_transform(self, img, keypoints):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        # resize the image to fit the crop size
        if st_size < self.c_size:
            rr = 1.0 * self.c_size / st_size
            wd = round(wd * rr)
            ht = round(ht * rr)
            st_size = 1.0 * min(wd, ht)
            img = img.resize((wd, ht), Image.BICUBIC)
            keypoints = keypoints * rr
        assert st_size >= self.c_size, print(wd, ht)
        assert len(keypoints) >= 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        if len(keypoints) > 0:
            keypoints = keypoints - [j, i]
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                       (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])

        gt_discrete = gen_discrete_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        assert np.sum(gt_discrete) == len(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
                keypoints[:, 0] = w - keypoints[:, 0] - 1
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
        gt_discrete = np.expand_dims(gt_discrete, 0)

        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(
            gt_discrete.copy()).float()