
import open3d as o3d
import argparse




def parse_args():
    parser = argparse.ArgumentParser(description='line_counter')

    parser.add_argument('--dir', type=str, default='',
                        help='Where the pointcloud is stored')

    args = parser.parse_args()
    return args


if __name__ == "__main__":  
    args = parse_args()
    file = open(args.dir)
    for i, l in enumerate(file):
        pass
    print(i + 1)
    

    




