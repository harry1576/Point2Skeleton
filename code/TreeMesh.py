import numpy as np 
import open3d as o3d
import argparse
from glob import glob
from shutil import copyfile
import os
import torch

import colourmap as colourmap

from structs.struct import to_structs, struct
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

def load(filename):
    return to_structs(np.load(filename, allow_pickle=True).item())


def flatten_parts(part_root):   
    parts = []

    def flatten(part):
        parts.append(part)
        for child in part.children:
            flatten(child)
            
    flatten(part_root)
    return parts

def get_random_points(array, number_of_points):
    return array[np.random.randint(len(array), size=(number_of_points))]


def merge_meshes(meshes):
    sizes = [mesh.vertices.shape[0] for mesh in meshes]
    offsets = np.cumsum([0] + sizes)

    part_indexes = np.repeat(np.arange(0, len(meshes)), sizes)
    
    mesh = struct(
        triangles = np.concatenate([mesh.triangles + offset for offset, mesh in zip(offsets, meshes)]),
        vertices = np.concatenate([mesh.vertices for mesh in meshes]),
        # uvs = np.concatenate([mesh.uvs for mesh in meshes]),
        part_indexes = part_indexes
    )
    return mesh


def build_mesh(tree, filter_classes = []):
    parts = flatten_parts(tree.parts)
    
    meshes = [merge_meshes(part.meshes) for part in parts if part.class_name not in filter_classes]
    merged = merge_meshes(meshes)

    instance_colors = colourmap.generate(len(parts))
    class_colors = colourmap.generate(len(tree.classes), cmap='Set1')

    classes = {k: i for i, k in enumerate(tree.classes)}   
    part_classes = [classes[part.class_name] for part in parts] 
    part_class_colors = class_colors[part_classes]

    return merged._extend(instance_colors = instance_colors, class_colors=part_class_colors)


def triangle_mesh(mesh_data, color_map):
    triangles = o3d.utility.Vector3iVector(mesh_data.triangles)
    vertices = o3d.utility.Vector3dVector(mesh_data.vertices)

    mesh = o3d.geometry.TriangleMesh(vertices, triangles)
    mesh.compute_vertex_normals()

    mesh.vertex_colors = o3d.utility.Vector3dVector(color_map[mesh_data.part_indexes])
    return mesh



if __name__ == "__main__":  

    args = parse_args()
    
    
    for folder in glob(args.pcd_read_dir + "/*"):  # Copy point clouds over from read directory and convert into ply ascii 
        if not os.path.isdir(args.data_write_dir+"/"+folder.split("/")[-1]):
                    
            os.mkdir(args.data_write_dir+"/"+folder.split("/")[-1], 0o755)
            pcd = o3d.io.read_point_cloud(f"{folder}/back_close/scan_file_integrated.pcd")
            #pcd = pcd.voxel_down_sample(voxel_size=0.01)
            #norm = np.linalg.norm(random_sample)
            #norm_random_sample = random_sample / norm
            random_sample = get_random_points(np.asarray(pcd.points),500000)
            pcd.points =  o3d.utility.Vector3dVector(random_sample)
            o3d.io.write_point_cloud(f"{args.data_write_dir}/{folder.split('/')[-1]}/{folder.split('/')[-1]}.ply", pcd, write_ascii=True,print_progress=True)
        
    
    for folder in glob(args.tree_dir + "/*"):   
        if not os.path.isdir(args.data_write_dir+"/"+folder.split("/")[-1]):
        
            filter=['Node']
            instance = False
            
            tree = load(str(folder))
            mesh = build_mesh(tree, filter)
            
            tri_mesh = triangle_mesh(mesh, color_map=mesh.part_colors if instance else mesh.class_colors)
            pcd = tri_mesh.sample_points_uniformly(500000)
            o3d.io.write_point_cloud(f"{args.data_write_dir}/{folder.split('/')[-1][:-5]}/{folder.split('/')[-1][:-5]}_skel.ply", pcd, write_ascii=True,print_progress=True)
        


   