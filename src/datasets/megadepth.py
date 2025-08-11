import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from kornia.geometry.epipolar import essential_from_Rt, fundamental_from_essential
from torch.utils.data import Dataset
from loguru import logger

from src.utils.dataset import read_megadepth_gray, read_megadepth_depth, scale_intrinsics


class MegaDepthDataset(Dataset):
    def __init__(self,
                 root_dir,
                 npz_path,
                 mode='train',
                 min_overlap_score=0.4,
                 img_resize=None,
                 df=None,
                 img_padding=False,
                 depth_padding=False,
                 augment_fn=None,
                 **kwargs):
        """
        Manage one scene(npz_path) of MegaDepth dataset.

        Args:
            root_dir (str): megadepth root directory that has `phoenix`.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            mode (str): options are ['train', 'val', 'test']
            min_overlap_score (float): how much a pair should have in common. In range of [0, 1]. Set to 0 when testing.
            img_resize (int, optional): the longer edge of resized images. None for no resize. 640 is recommended.
                                        This is useful during training with batches and testing with memory intensive algorithms.
            df (int, optional): image size division factor. NOTE: this will change the final image size after img_resize.
            img_padding (bool): If set to 'True', zero-pad the image to squared size. This is useful during training.
            depth_padding (bool): If set to 'True', zero-pad depthmap to (2000, 2000). This is useful during training.
            augment_fn (callable, optional): augments images with pre-defined visual effects.
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.scene_id = npz_path.split('.')[0]

        # prepare scene_info and pair_info
        if mode == 'test' and min_overlap_score != 0:
            logger.warning("You are using `min_overlap_score`!=0 in test mode. Set to 0.")
            min_overlap_score = 0
        self.scene_info = np.load(npz_path, allow_pickle=True)
        self.pair_infos = self.scene_info['pair_infos'].copy()
        del self.scene_info['pair_infos']
        self.pair_infos = [pair_info for pair_info in self.pair_infos if pair_info[1] > min_overlap_score]
        #self.pair_infos = [pair_info for pair_info in self.pair_infos if pair_info[1] < 0.3]
        # parameters for image resizing, padding and depthmap padding
        if mode == 'train':
            assert img_resize is not None and img_padding and depth_padding
        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding
        self.depth_max_size = 2000 if depth_padding else None  # the upperbound of depthmaps size in megadepth.

        # for training LoFTR
        self.augment_fn = augment_fn if mode == 'train' else None
        self.coarse_scale = getattr(kwargs, 'coarse_scale', 0.125)

    def __len__(self):
        return len(self.pair_infos)

    def __getitem__(self, idx):
        (idx0, idx1), overlap_score, central_matches = self.pair_infos[idx]

        # read grayscale image and mask. (1, h, w) and (h, w)
        img0_name = self.scene_info['image_paths'][idx0].split('/')[-1]
        img0_dir = osp.dirname(osp.dirname(self.scene_info['depth_paths'][idx0]))
        img0_path = osp.join(img0_dir, 'imgs', img0_name)
        img1_name = self.scene_info['image_paths'][idx1].split('/')[-1]
        img1_dir = osp.dirname(osp.dirname(self.scene_info['depth_paths'][idx1]))
        img1_path = osp.join(img1_dir, 'imgs', img1_name)

        if 'test' in self.root_dir:
            img_name0 = osp.join(self.root_dir, self.scene_info['image_paths'][idx0])
            img_name1 = osp.join(self.root_dir, self.scene_info['image_paths'][idx1])
        else:
            img_name0 = osp.join(self.root_dir, img0_path)
            img_name1 = osp.join(self.root_dir, img1_path)

        # TODO: Support augmentation & handle seeds for each worker correctly.
        #print(img_name0, img_name1)
        image0, mask0, scale0, shape0 = read_megadepth_gray(
            img_name0, self.img_resize, self.df, self.img_padding, None)
            # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
        image1, mask1, scale1, shape1 = read_megadepth_gray(
            img_name1, self.img_resize, self.df, self.img_padding, None)
            # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))

        # read depth. shape: (h, w)
        if self.mode in ['train', 'val']:
            depth0 = read_megadepth_depth(
                osp.join(self.root_dir, self.scene_info['depth_paths'][idx0]), pad_to=self.depth_max_size)
            depth1 = read_megadepth_depth(
                osp.join(self.root_dir, self.scene_info['depth_paths'][idx1]), pad_to=self.depth_max_size)
        else:
            depth0 = depth1 = torch.tensor([])

        # read intrinsics of original size
        K_0 = torch.tensor(self.scene_info['intrinsics'][idx0].copy(), dtype=torch.float).reshape(3, 3)
        K_1 = torch.tensor(self.scene_info['intrinsics'][idx1].copy(), dtype=torch.float).reshape(3, 3)
        scale_factor_0 = (self.coarse_scale / scale0[0], self.coarse_scale / scale0[1])
        scale_factor_1 = (self.coarse_scale / scale1[0], self.coarse_scale / scale1[1])
        K_0_c = scale_intrinsics(K_0[None], scale_factor=scale_factor_0)[0]
        K_1_c = scale_intrinsics(K_1[None], scale_factor=scale_factor_1)[0]

        # read and compute relative poses
        T0 = self.scene_info['poses'][idx0]
        T1 = self.scene_info['poses'][idx1]
        T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)[:4, :4]  # (4, 4)
        T_1to0 = T_0to1.inverse()
        Em = essential_from_Rt(
            torch.tensor(T0[:3, :3][None], dtype=torch.float32), 
            torch.tensor(T0[:3, 3][None, ..., None], dtype=torch.float32), 
            torch.tensor(T1[:3, :3][None], dtype=torch.float32), 
            torch.tensor(T1[:3, 3][None, ..., None], dtype=torch.float32)
        )[0]
        Fm = fundamental_from_essential(
            Em[None],
            K_0[None],
            K_1[None]
        )[0]

        data = {
            'image0': image0,  # (1, h, w)
            'depth0': depth0,  # (h, w)
            'image1': image1,
            'depth1': depth1,
            'T_0to1': T_0to1,  # (4, 4)
            'T_1to0': T_1to0,
            'K0': K_0,  # (3, 3)
            'K1': K_1,
            'K0_c': K_0_c,  # (3, 3)
            'K1_c': K_1_c,
            'Em': Em,
            'Fm': Fm,
            'scale0': scale0,  # [scale_w, scale_h]
            'scale1': scale1,
            'dataset_name': 'MegaDepth',
            'scene_id': self.scene_id,
            'pair_id': idx,
            'pair_idxs': (idx0, idx1),
            'img_shape0': shape0,
            'img_shape1': shape1,
            'pair_names': (self.scene_info['image_paths'][idx0], self.scene_info['image_paths'][idx1]),
            'img_paths': (img_name0, img_name1)
        }

        # for LoFTR training
        if mask0 is not None:  # img_padding is True
            if self.coarse_scale:
                [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                       scale_factor=self.coarse_scale,
                                                       mode='nearest',
                                                       recompute_scale_factor=False)[0].bool()
            data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})

        return data
