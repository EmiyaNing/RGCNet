from __future__ import absolute_import, division, print_function
import os
import argparse

from tester import Tester
#from test import Tester

parser = argparse.ArgumentParser(description="360 Degree Panorama Depth Estimation Training")

# system settings
parser.add_argument("--num_workers", type=int, default=8, help="number of dataloader workers")
parser.add_argument("--gpu_devices", type=int, nargs="+", default=[0], help="available gpus")

# model settings
parser.add_argument("--impact_factor", type=float, default=2, help="To decide the impact of background depth estimation")

# eval setting settings
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--eval_all", action="store_true", help="eval all training models.")
parser.add_argument("--start_epoch", type=int, default=0, help="decide which epoch start to eval")

# loading and logging settings
parser.add_argument("--load_weights_dir", default='./tmp_st3d/panobackbone_st3d/models', type=str, help="folder of model to load")#, default='./tmp_abl_offset/panodepth/models/weights_49'
parser.add_argument("--log_dir", type=str, default=os.path.join(os.path.dirname(__file__), "tmp"), help="log directory")
parser.add_argument("--log_frequency", type=int, default=100, help="number of batches between each tensorboard log")

args = parser.parse_args()


def main():

    tester = Tester(args)
    tester.test()


if __name__ == "__main__":
    main()
