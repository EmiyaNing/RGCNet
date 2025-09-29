from __future__ import absolute_import, division, print_function
import os
import re


import tqdm
import torch

import numpy as np

from torch.utils.data import DataLoader

import post_proc 

from metrics import Evaluator
from RGCNet import RGCNet
from structured3d import Structured3D
from layout_utils import *


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


class Tester:
    def __init__(self, settings):
        self.settings = settings

        self.device = torch.device("cuda" if len(self.settings.gpu_devices) else "cpu")
        self.gpu_devices = ','.join([str(id) for id in settings.gpu_devices])
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_devices

        self.log_path = os.path.join(self.settings.log_dir, self.settings.model_name)

        val_dataset = Structured3D('.data/structured3d/','./structured3d_test.txt', 
                                    disable_color_augmentation=False,
                                    disable_LR_filp_augmentation=False,
                                    disable_yaw_rotation_augmentation=False, 
                                    is_training=False)
        
        self.test_loader = DataLoader(val_dataset, self.settings.batch_size, False,
                                     num_workers=self.settings.num_workers, pin_memory=True, drop_last=True)
        

        self.model = RGCNet()
   
        self.evaluator = Evaluator()
        self.model.to(self.device)
        self.model.eval()


    def post_process_layout(self, boundry, corner):
        H = 512
        W = 1024
        #pdb.set_trace()
        boundry = boundry.detach().cpu().numpy()
        corner  = corner.sigmoid().detach().cpu().numpy()

        boundry = (boundry[0] / np.pi + 0.5) * H - 0.5
        boundry[0] = np.clip(boundry[0], 1, H/2 - 1)
        boundry[1] = np.clip(boundry[1], H/2+1, H-2)

        corner  = corner[0, 0]
        z0 = 50

        #pdb.set_trace()
        _, z1 = post_proc.np_refine_by_fix_z(*boundry, z0)

        r = int(round(W * 0.05 / 2))

        xs_ = find_N_peaks(corner, r=r, min_v=0, N=None)[0]
        cor, xy_cor = post_proc.gen_ww(xs_, boundry[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=False)

        cor = np.hstack([cor, post_proc.infer_coory(cor[:, 1], z1 - z0, z0)[:, None]])
        cor_id = np.zeros((len(cor) * 2, 2), np.float32)
        for j in range(len(cor)):
            cor_id[j*2] = cor[j, 0], cor[j, 1]
            cor_id[j*2 + 1] = cor[j, 0], cor[j, 2]

        # Normalized to [0, 1]
        cor_id[:, 0] /= W
        cor_id[:, 1] /= H
        return cor_id


    def generate_backgroud_depth(self, cor_id, w, h, camera_height):
        cor_id[:, 0] = cor_id[:, 0] * w
        cor_id[:, 1] = cor_id[:, 1] * h
        vc, vf = cor_2_1d(cor_id, h, w)
        vc = vc[None, :]  # [1, w]
        vf = vf[None, :]  # [1, w]
        assert (vc > 0).sum() == 0
        assert (vf < 0).sum() == 0

        # Per-pixel v coordinate (vertical angle)
        vs = ((np.arange(h) + 0.5) / h - 0.5) * np.pi
        vs = np.repeat(vs[:, None], w, axis=1)  # [h, w]

        # Floor-plane to depth
        floor_h = camera_height
        floor_d = np.abs(floor_h / np.sin(vs))

        # wall to camera distance on horizontal plane at cross camera center
        cs = floor_h / np.tan(vf)

        # Ceiling-plane to depth
        ceil_h = np.abs(cs * np.tan(vc))      # [1, w]
        ceil_d = np.abs(ceil_h / np.sin(vs))  # [h, w]

        # Wall to depth
        wall_d = np.abs(cs / np.cos(vs))  # [h, w]

        # Recover layout depth
        floor_mask = (vs > vf)
        ceil_mask = (vs < vc)
        wall_mask = (~floor_mask) & (~ceil_mask)
        depth = np.zeros([h, w], np.float32)    # [h, w]
        depth[floor_mask] = floor_d[floor_mask]
        depth[ceil_mask] = ceil_d[ceil_mask]
        depth[wall_mask] = wall_d[wall_mask]

        assert (depth == 0).sum() == 0

        return depth


    def test(self):
        eval_all = self.settings.eval_all
        if eval_all:
            self.validate_all_ckpt()
        else:
            self.validate_one_ckpt()


    def process_batch(self, inputs):
        for key, ipt in inputs.items():
            if key not in ["rgb", "cur_name"]:
                inputs[key] = ipt.to(self.device)

        equi_inputs = inputs["normalized_rgb"]# * inputs["val_mask"]

        # cube_inputs = inputs["normalized_cube_rgb"]

        outputs = self.model(equi_inputs)
        if self.settings.RGCNet:
            corner  = outputs['corner']
            boundry = outputs['boundry']

            cor_id  = self.post_process_layout(boundry, corner)
            #pdb.set_trace()
            back_depth = self.generate_backgroud_depth(cor_id, 1024, 512, inputs['camera_z'].cpu().numpy() / 1000)
            background = torch.tensor(back_depth, device=inputs['camera_z'].device)
            outputs['background'] = background

            weights    = outputs['back_segm'].sigmoid() / self.settings.impact_factor
            fusion     = weights * background + (1 - weights) * outputs['pred_depth']
            #pdb.set_trace()
            outputs['fusion_depth'] = fusion
        return outputs


    def validate_one_ckpt(self):
        ckpt_file = self.settings.load_weights_dir
        self.load_model_ckpt(ckpt_file)
        self.evaluator.reset_eval_metrics()

        pbar = tqdm.tqdm(self.test_loader)
        pbar.set_description("Testing single ckpt file")

        for batch_idx, inputs in enumerate(pbar):
            outputs    = self.process_batch(inputs)
            gt_depth   = inputs["gt_depth"] * inputs["val_mask"]             
            fusion_depth = outputs['fusion_depth'] * inputs["val_mask"]
            self.evaluator.compute_eval_metrics(gt_depth.detach().cpu(), fusion_depth.detach().cpu())

        self.evaluator.print()
        del outputs, inputs


    def validate_all_ckpt(self):
        
        self.evaluator.reset_eval_metrics()
        if self.settings.RGCNet:
            self.evaluator_fusion.reset_eval_metrics()
        

        file_dirs = self.settings.load_weights_dir
        file_lists= os.listdir(file_dirs)
        file_lists= sorted(file_lists, key=natural_sort_key)
        file_lists= file_lists[1:]
        
        start_epoch = self.settings.start_epoch
        file_lists  = file_lists[start_epoch:]



        for idx, ckpt_file in enumerate(file_lists):
            ckpt_file_path = os.path.join(file_dirs, ckpt_file)
            self.load_model_ckpt(ckpt_file_path)

            pbar = tqdm.tqdm(self.test_loader)
            pbar.set_description("Testing single ckpt file")
            print("Start to eval epoch: ", idx + start_epoch)

            for batch_idx, inputs in enumerate(pbar):
                outputs    = self.process_batch(inputs)
                gt_depth   = inputs["gt_depth"] * inputs["val_mask"]             
                fusion_depth = outputs['fusion_depth'] * inputs["val_mask"]
                self.evaluator.compute_eval_metrics(gt_depth.detach().cpu(), fusion_depth.detach().cpu())
            
            self.evaluator.print()
            pbar.reset()
            del outputs, inputs



    def load_model_ckpt(self, ckpt_dirs):
        """Load model from disk
        """
        load_ckpt_weights_dir = os.path.expanduser(ckpt_dirs)

        assert os.path.isdir(load_ckpt_weights_dir), \
            "Cannot find folder {}".format(load_ckpt_weights_dir)
        print("loading model from folder {}".format(load_ckpt_weights_dir))

        path = os.path.join(load_ckpt_weights_dir, "{}.pth".format("model"))
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
