import os
import torch
import numpy as np
import gc
import matplotlib.pyplot as plt

import utils.general as utils
from utils import rend_util


class SceneDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 expname,
                 img_res,
                 fg_ratio=0,
                 reliable_only=False,
                 ):

        self.fg_ratio = float(fg_ratio)
        self.reliable_only = reliable_only
        self.virtual_sculpting = True

        # set directory
        self.instance_dir = os.path.join('../data', f'{expname}')
        assert os.path.exists(self.instance_dir), "Data directory is empty"
        self.sampling_idx = None

        # set parameters
        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res

        # rgb images
        image_dir = f"{self.instance_dir}/input/image/"
        image_paths = sorted(utils.glob_imgs(image_dir))
        self.n_frames = 1  # single frame
        self.n_images = len(image_paths)

        self.rgb_images = []
        for path in image_paths:
            rgb = rend_util.load_rgb(path)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())

        # kaleidoscopic mask
        mask_dir = f"{self.instance_dir}/input/mask_kaleidoscope/"
        mask_paths = sorted(utils.glob_imgs(mask_dir))

        self.kaleidoscopic_mask = []
        for path in mask_paths:
            kaleidoscopic_mask = rend_util.load_mask(path)
            kaleidoscopic_mask = kaleidoscopic_mask.reshape(-1)
            self.kaleidoscopic_mask.append(torch.from_numpy(kaleidoscopic_mask).bool())

        # camera pose
        self.cam_file = f'{self.instance_dir}/input/cameras.npz'
        camera_dict = np.load(self.cam_file)
        self.n_images_total = camera_dict['num_camera']

        K_list = [camera_dict['K_%d' % idx].astype(np.float32) for idx in range(self.n_images_total)]
        R_list = [camera_dict['R_%d' % idx].astype(np.float32) for idx in range(self.n_images_total)]
        C_list = [camera_dict['C_%d' % idx].astype(np.float32) for idx in range(self.n_images_total)]
        self.label_list = [camera_dict['label_%d' % idx].astype(np.int64) for idx in range(self.n_images_total)]

        self.intrinsics_all = []
        self.pose_all = []
        for K, R, C in zip(K_list, R_list, C_list):
            self.intrinsics_all.append(torch.from_numpy(K).float())
            pose = np.eye(4)
            pose[:3, :3] = R.transpose()
            pose[:3, 3:4] = C
            self.pose_all.append(torch.from_numpy(pose).float())

        # visual hull        
        self.vh_num_bounce = torch.from_numpy(camera_dict['visualhull_num_bounce']).int().reshape(-1)
        self.mirror_sequence_vh = torch.from_numpy(camera_dict['mirror_sequence_vh']).int().reshape(-1)
        # self.num_bounces_frame0 = self.vh_num_bounces.clone()

        # mirror sequence edge
        self.mirror_sequence_edge = torch.from_numpy(camera_dict['mirror_sequence_edge']).bool().reshape(-1)
        plt.imsave(f'debug/mirror_sequence_edge.png', self.mirror_sequence_edge.reshape(self.img_res[0], self.img_res[1]).numpy())

        # mirror_pixel
        self.mirror_pixel = torch.from_numpy(camera_dict['mirror_pixel']).bool().reshape(-1)
        plt.imsave(f'debug/mirror_pixel.png', self.mirror_pixel.reshape(self.img_res[0], self.img_res[1]).numpy())

        # is_unreliable
        self.reliable = ~torch.from_numpy(camera_dict['is_unreliable']).bool().reshape(-1)
        self.reliable_concat = self.reliable.reshape(self.img_res).repeat(self.n_images_total, 1).reshape(-1)
        utils.mkdir_ifnotexists('debug')
        plt.imsave(f'debug/img_reliable.png', ((self.rgb_images[0] + 1) / 2 * self.reliable.reshape(-1, 1).repeat(1, 3)).reshape(self.img_res[0], self.img_res[1], 3).numpy())

        # bounce_hit_visualhull
        self.bounce_hit_visualhull = torch.from_numpy(camera_dict['bounce_hit_visualhull']).bool().reshape(self.total_pixels, -1)
        plt.imsave(f'debug/bounce_hit_visualhull0.png', self.bounce_hit_visualhull[:, 0].reshape(self.img_res[0], self.img_res[1]).numpy())

        # mirror
        self.n_mirrors = 4
        mirror_n = [camera_dict['mirror_n_%d' % idx].astype(np.float32) for idx in range(self.n_mirrors)]
        mirror_d = [camera_dict['mirror_d_%d' % idx].astype(np.float32) for idx in range(self.n_mirrors)]

        self.mirror_n = []
        self.mirror_d = []
        for mirror_n, mirror_d in zip(mirror_n, mirror_d):
            self.mirror_n.append(torch.from_numpy(mirror_n).float())
            self.mirror_d.append(torch.from_numpy(mirror_d).float())

        # gt
        if 'gt_num_bounce' in camera_dict.keys():
            self.gt_num_bounce = torch.from_numpy(camera_dict['gt_num_bounce']).int()
        else:
            self.gt_num_bounce = self.vh_num_bounce

        # idx eval
        self.idx_eval = 0

        # set frame 0
        self.frame_now = 0

        # uv
        self.uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        self.uv = torch.from_numpy(np.flip(self.uv, axis=0).copy())
        self.uv = self.uv.reshape(2, -1).transpose(1, 0)


        is_fg = self.kaleidoscopic_mask[0] & ~self.mirror_sequence_edge
        is_bg = ~self.kaleidoscopic_mask[0] & ~self.mirror_sequence_edge
        if self.reliable_only:
            is_fg = is_fg & self.reliable
        self.idx_fg = torch.nonzero(is_fg).squeeze()
        self.idx_bg = torch.nonzero(is_bg).squeeze()
        self.idx_fgbg = torch.cat((self.idx_fg.reshape(-1), self.idx_bg.reshape(-1)), dim=0)

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        # evaluation
        if self.sampling_idx is None:
            idx = 0
            sample = {
                "object_mask": self.kaleidoscopic_mask[idx],
                "uv": self.uv.clone(),
                "intrinsics": self.intrinsics_all[idx],
                "pose": self.pose_all[idx],
            }

            ground_truth = {
                "rgb": self.rgb_images[idx],
                "mirror_sequence_edge": self.mirror_sequence_edge,
                "vh_num_bounce": self.vh_num_bounce,
                "gt_num_bounce": self.gt_num_bounce,
                "mirror_pixel": self.mirror_pixel,
                "bounce_hit_visualhull": self.bounce_hit_visualhull,
            }

        # training
        else:
            idx = 0  # use real camera
            sampling_size = self.sampling_idx.shape[0]
            self.update_sampling_idx(sampling_size)

            u = self.sampling_idx % self.img_res[1]
            v = self.sampling_idx // self.img_res[1]
            uv = torch.stack((u, v), dim=1)

            sample = {
                "object_mask": self.kaleidoscopic_mask[idx][self.sampling_idx],
                "uv": uv.clone(),
                "intrinsics": self.intrinsics_all[idx],
                "pose": self.pose_all[idx],
            }

            ground_truth = {
                "rgb": self.rgb_images[idx][self.sampling_idx, :],
                "mirror_sequence_edge": self.mirror_sequence_edge[self.sampling_idx],
                "vh_num_bounce": self.vh_num_bounce[self.sampling_idx],
                "gt_num_bounce": self.gt_num_bounce[self.sampling_idx],
                "cam_num_bounce": torch.ones_like(self.gt_num_bounce[self.sampling_idx]),
                "bounce_hit_visualhull": self.bounce_hit_visualhull[self.sampling_idx, :],
            }

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def update_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            if self.fg_ratio == 0:
                # random sample
                n_fgbg = self.idx_fgbg.shape[0]
                randint_fgbg = torch.randint(low=0, high=n_fgbg, size=(sampling_size,))
                self.sampling_idx = self.idx_fgbg[randint_fgbg]
            else:
                # fg bg sample
                idx_fg = self.idx_fg.clone()
                idx_bg = self.idx_bg.clone()
                n_fg = round(sampling_size * self.fg_ratio)
                n_bg = sampling_size - n_fg
                randint_fg = torch.randint(low=0, high=idx_fg.shape[0], size=(n_fg,))
                randint_bg = torch.randint(low=0, high=idx_bg.shape[0], size=(n_bg,))
                self.sampling_idx = torch.cat((idx_fg[randint_fg], idx_bg[randint_bg]), dim=0)

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']

    def get_n_frames(self):
        return self.n_frames

    def change_frame(self, frame):
        self.frame_now = 0  # frame_now is not used
        # self.frame_now = frame  # frame_now is not used

    def get_n_images_total(self):
        return self.n_images_total

    def get_mirror(self):
        return self.mirror_n, self.mirror_d

    def get_label_list(self):
        return self.label_list
