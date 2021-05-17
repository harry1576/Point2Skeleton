import torch

e = 0.57735027
sample_directions = torch.tensor([[e, e, e], [e, e, -e], [e, -e, e], [e, -e, -e], [-e, e, e], [-e, e, -e], [-e, -e, e], [-e, -e, -e]]).double() 


skel_gt_xyz = torch.as_tensor([1,1,1]).double() 
radius = torch.as_tensor(2).double() 

skel_gt_xyz = torch.repeat_interleave(skel_gt_xyz, 8, dim=0).reshape(1,8,-1) 


y = skel_gt_xyz + radius * sample_directions
x = skel_gt_xyz + sample_directions * radius

print(y)
print(x)