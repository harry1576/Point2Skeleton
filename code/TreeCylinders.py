import numpy as np 
import open3d as o3d
import argparse
from glob import glob
from shutil import copyfile
import os
import torch

from structs.struct import to_structs
import operator
import functools

def parse_args():
    parser = argparse.ArgumentParser(description='Vine2PCD')

    parser.add_argument('--pcd_read_dir', type=str, default='/local/Point2Skeleton/data/renders',
                        help='Where the pcd are stored')
    
    parser.add_argument('--tree_dir', type=str, default='/local/Point2Skeleton/data/trees',
                        help='Where trees are stored')

    parser.add_argument('--data_write_dir', type=str, default='/local/Point2Skeleton/data/pointclouds',
                        help='Where to write back the ply files')
    
    args = parser.parse_args()
    return args


def flatten_parts(part_root):   
    parts = []

    def flatten(part):
        parts.append(part)
        for child in part.children:
            flatten(child)
            
    flatten(part_root)
    return parts


def reduce_add(xs):
    return functools.reduce(operator.add, xs)


def key_points(spine):
    
    d = spine.points[1:] - spine.points[:-1]
    norm_d = np.linalg.norm(d, 2, axis=1) / (2 * spine.radius[:-1])
    t = np.floor(norm_d.cumsum())
    _, inds = np.unique(t, return_index=True)
    
    return inds


def sphere_points(spines):
    """ Takes spines and outputs a np array of xyz coordinates"""

    meshes = []
    
    for spine in spines:
        for i in range(0,len(spine.points)-1):
       
            direction_vector = (spine.points[i] - spine.points[i+1])

            xy_angle =  np.arctan2(direction_vector[0],direction_vector[1])           
            xz_angle =  np.arctan2(direction_vector[0],direction_vector[2])           
            yz_angle =  np.arctan2(direction_vector[1],direction_vector[2])           

            height = np.linalg.norm(spine.points[i] - spine.points[i+1])

            mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=spine.radius[i],height=height)
            R = mesh.get_rotation_matrix_from_xyz((-yz_angle,xz_angle,-xy_angle))
            
            mesh = mesh.rotate(R, center=(0, 0, 0))     
            
            meshes.append(mesh.translate(spine.points[i]))
            
    return meshes
        
        
        

def get_random_points(array, number_of_points):
    return array[np.random.randint(len(array), size=(number_of_points))]


if __name__ == "__main__":  

    args = parse_args()
    
    
    for folder in glob(args.tree_dir + "/*"):   
        tree = to_structs(np.load(str(folder), allow_pickle=True).item())
        parts = flatten_parts(tree.parts)
        spines = [part.spine for part in parts if part.class_name != 'Node']           
        meshes = sphere_points(spines)
        #meshes.compute_vertex_normals()
        

        o3d.visualization.draw_geometries(meshes)
        quit()

        #pcd = o3d.geometry.PointCloud()
        
        #norm = np.linalg.norm(points)
        #norm_point = points / norm
        
        #pcd.points = o3d.utility.Vector3dVector(points)
        #o3d.io.write_point_cloud(f"{args.data_write_dir}/{folder.split('/')[-1][:-5]}/{folder.split('/')[-1][:-5]}_skel.ply", pcd, write_ascii=True,print_progress=True)

        

   