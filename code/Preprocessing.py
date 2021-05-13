import numpy as np 
import open3d as o3d
import argparse
from glob import glob
from shutil import copyfile
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Vine2PCD')

    parser.add_argument('--data_read_dir', type=str, default='data/renders',
                        help='Where the renders are stored')

    parser.add_argument('--data_write_dir', type=str, default='data/pointclouds',
                        help='Where to write back the ply files')

    args = parser.parse_args()
    return args


if __name__ == "__main__":  

    args = parse_args()
    for folder in glob(args.data_read_dir + "/*"):  # Copy point clouds over from read directory and convert into ply ascii 
        if not os.path.isdir(args.data_write_dir+"/"+folder.split("/")[-1]):
          os.mkdir(args.data_write_dir+"/"+folder.split("/")[-1], 0o755)
          pcd = o3d.io.read_point_cloud(f"{folder}/back_close/scan_file_integrated.pcd")
          #pcd = pcd.voxel_down_sample(voxel_size=0.01)
          o3d.io.write_point_cloud(f"{args.data_write_dir}/{folder.split('/')[-1]}/{folder.split('/')[-1]}.ply", pcd, write_ascii=True,print_progress=True)

        
        
        



