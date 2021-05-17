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

    surface_points = []
    
    for spine in spines:
    
        points = key_points(spine)
        
        e = 0.57735027
        sample_directions = torch.tensor([[e, e, e], [e, e, -e], [e, -e, e], [e, -e, -e], [-e, e, e], [-e, e, -e], [-e, -e, e], [-e, -e, -e]]).double()   
        #sample_directions = torch.unsqueeze(sample_directions, 0)
        
        sample_directions = sample_directions.repeat(len(points), 1)
        skel_gt_xyz = torch.as_tensor([spine.points[i] for i in points])
        skel_gt_xyz = torch.repeat_interleave(skel_gt_xyz, 8, dim=0)
       
        skel_gt_radius =  torch.as_tensor([spine.radius[i] for i in points]).double()
        skel_gt_radius = torch.repeat_interleave(skel_gt_radius, 8, dim=0).reshape(-1,1)

        sample_points = skel_gt_xyz + skel_gt_radius * sample_directions

        surface_points = np.append(surface_points,sample_points.numpy())
    
    surface_points = np.reshape(surface_points,(-1,3))
    return surface_points

    if 1 == 1:
        x = skel_gt_radius * sample_directions
        print(x)
        return x.numpy() 
        
        #print(sample_directions)
        quit()
        
        skel_gt_xyz = torch.as_tensor([spine.points[i] for i in points])
        skel_gt_xyz = torch.repeat_interleave(skel_gt_xyz, 8, dim=0).reshape(-1,8)
        skel_gt_radius =  torch.as_tensor([spine.radius[i] for i in points])
        skel_gt_radius = torch.repeat_interleave(skel_gt_radius, 8, dim=0).reshape(-1,8)
        print(skel_gt_radius)
        print(skel_gt_xyz)

        
        sample_points = skel_gt_xyz.double() + skel_gt_radius.double() # add radius to each point
        
        
        
        sample_points = torch.repeat_interleave(sample_points, 8, dim=0).reshape(len(points),8,-1) 
        sample_points = sample_points * sample_directions
        
        print(sample_points)
        #sample_points = sample_points.reshape(-1,3)
        
        #sample_points = skel_gt_xyz
        
        #print(skel_gt_xyz)
        
        surface_points = np.append(surface_points,sample_points.numpy())
        
        
    surface_points = np.reshape(surface_points,(-1,3))
    #surface_points.extend(list(sample_points.numpy()))
    #print(surface_poiqnts)
    return surface_points

    #print(skel_gt_xyz)
    print("xyz")
    print(skel_gt_xyz)
    print("r")
    print(skel_gt_radius)   
    print("yeet")
    
    

    
    sample_centers = torch.repeat_interleave(skel_gt_xyz, 8, dim=1)
    sample_radius = torch.repeat_interleave(skel_gt_radius, 3, dim=0)

    #sample_xyz = sample_centers + sample_radius
    
    
    print(sample_centers.shape)
    print(sample_radius.shape)
    #print(sample_xyz)

    quit()
    #sample_centers = torch.repeat_interleave(skel_gt_xyz, 8, dim=1)
    #sample_radius = torch.repeat_interleave(skel_gt_radius, 8, dim=1)
    
    print(sample_centers)
    quit()
    #print(sample_radius.shape)

    
    sample_xyz = sample_centers + sample_radius # * sample_directions
    
    #sample_centers = torch.repeat_interleave(skel_xyz, 8, dim=1)
    #print(sample_xyz)

    #returns [[x,y,z],r] for each pt
        
    #x,y,z = [spine.points[i] for i in points]
    #print(x,y,z,r)
    
    #xyzr = np.copy(spine.points)

   # print(xyzr)
    #print(spine.points[0].append(spine.radius[0]))
    #for point in points:
    #    print(point)
        #xyzr = np.append(xyzr[point],spine.points[point]) 
    #print(xyzr)
    #print([(spine.points[i],spine.radius[i]) for i in points])
    #return [spine.points[i] for i in points], [spine.radius[i] for i in points]
    
    points = key_points(spine)

    return [(spine.points[i],spine.radius[i]) for i in points]

    return [o3d.geometry.TriangleMesh.create_sphere(radius=spine.radius[i], resolution=6).translate(spine.points[i]) for i in points]



if __name__ == "__main__":  

    args = parse_args()
    
    
    #Get Stereo Meshes
    """
    for folder in glob(args.pcd_read_dir + "/*"):  # Copy point clouds over from read directory and convert into ply ascii 
        if not os.path.isdir(args.data_write_dir+"/"+folder.split("/")[-1]):
          os.mkdir(args.data_write_dir+"/"+folder.split("/")[-1], 0o755)
          pcd = o3d.io.read_point_cloud(f"{folder}/back_close/scan_file_integrated.pcd")
          #pcd = pcd.voxel_down_sample(voxel_size=0.01)
          o3d.io.write_point_cloud(f"{args.data_write_dir}/{folder.split('/')[-1]}/{folder.split('/')[-1]}.ply", pcd, write_ascii=True,print_progress=True)
    """
    

    
    for folder in glob(args.tree_dir + "/*"):   
        print(folder)  
        tree = to_structs(np.load(str(folder), allow_pickle=True).item())
        parts = flatten_parts(tree.parts)
        spines = [part.spine for part in parts if part.class_name != 'Node']           
        
        points = sphere_points(spines)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()
        
        
        #point = points.reshape(3,-1)
         #print('ye')
        #print(points)
        
        #meshes = reduce_add(points)
        #meshes.compute_vertex_normals()
        
        
        
        #pcd = meshes.sample_points_uniformly(number_of_points=2500)
        #o3d.io.write_point_cloud(f"{args.data_write_dir}/{folder.split('/')[-1][:-5]}/{folder.split('/')[-1][:-5]}_skel.ply", pcd, write_ascii=True,print_progress=True)

        #o3d.io.write_point_cloud(f"{args.data_write_dir}/{folder.split('/')[-1]}/{folder.split('/')[-1]}.ply", pcd, write_ascii=True,print_progress=True)
        #vis = o3d.visualization.VisualizerWithKeyCallback()
        #vis.create_window()
        #vis.add_geometry(pcd)
        #vis.run()
    
    
    
    
    # Skel point cloud  old version
    """
    for folder in glob(args.tree_dir + "/*"):   
        print(folder)  
        tree = to_structs(np.load(str(folder), allow_pickle=True).item())
        parts = flatten_parts(tree.parts)
        spines = [part.spine for part in parts if part.class_name != 'Node']     
        points = [reduce_add(sphere_points(spine)) for spine in spines]
        meshes = reduce_add(points)
        meshes.compute_vertex_normals()
        #pcd = meshes.sample_points_uniformly(number_of_points=2500)
        o3d.visualization.draw_geometries([meshes])
        #o3d.io.write_point_cloud(f"{args.data_write_dir}/{folder.split('/')[-1][:-5]}/{folder.split('/')[-1][:-5]}_skel.ply", meshes, write_ascii=True,print_progress=True)

        #o3d.io.write_point_cloud(f"{args.data_write_dir}/{folder.split('/')[-1]}/{folder.split('/')[-1]}.ply", pcd, write_ascii=True,print_progress=True)
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()
    
    """

    
    
   
        
        



