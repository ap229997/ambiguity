import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt

class Parallelopiped(nn.Module):
    def __init__(self, xy_size=1):
        super().__init__()
        self.x_axis = torch.from_numpy(np.array([[1,0,0]])).float()
        self.y_axis = torch.from_numpy(np.array([[0,1,0]])).float()
        corners = np.array([
            [0,0,0.5], [1,0,0.5], [1,1,0.5], [0,1,0.5],
            [0,0,1.5], [1,0,1.5], [1,1,1.5], [0,1,1.5],
        ])
        self.corners = torch.from_numpy(corners).float() - 0.5
        self.xy_size = xy_size

    def forward(self, beta, translation):
        """
        Returns the corners of the parallelopiped with shape and pose parameters.
        beta: shape parameters (3,) gamma and angles
        theta: pose parameters (3,) 3D translation
        """
        r, theta, phi = beta[:,:1], beta[:,1:2], beta[:,2:3]
        x = r.abs() * torch.cos(theta) * torch.sin(phi)
        y = r.abs() * torch.sin(theta) * torch.sin(phi)
        z = r.abs() * torch.cos(phi)
        z_axis = torch.concat([x,y,z], dim=1)
        x_axis = self.x_axis * self.xy_size
        y_axis = self.y_axis * self.xy_size
        
        # B x 8 x 3
        corners = torch.stack([
            x_axis * x + y_axis * y + z_axis * z
            for x,y,z in self.corners
        ], dim=1)
        corners += translation[:,None,:]
        return corners, z_axis
    
    def to(self, device):
        self.x_axis = self.x_axis.to(device)
        self.y_axis = self.y_axis.to(device)
        self.corners = self.corners.to(device)
        return self

    def __repr__(self):
        return f"Parallelopiped()"

class Project(nn.Module):
    def __init__(self, f=1, cx=0., cy=0.):
        super().__init__()
        self.f = f
        self.cx = cx
        self.cy = cy

    def forward(self, corners):
        """
        corners: B x 8 x 3
        """
        x, y, z = corners[:,:,0], corners[:,:,1], corners[:,:,2]
        x = x / z * self.f + self.cx
        y = y / z * self.f + self.cy
        # x = x / (z + 1e-8) * self.f + self.cx
        # y = y / (z + 1e-8) * self.f + self.cy
        return torch.stack([x,y], dim=2)

    def __repr__(self):
        return f"Project(f={self.f}, cx={self.cx}, cy={self.cy})"
    
    
def vis_corners(ax, corners_2d, col='r', H=112, W=112):
    edges = [[0,1], [1,2], [2,3], [3,0], [4,5], [5,6], [6,7], [7,4], [0,4], [1,5], [2,6], [3,7]]
    for i, e in enumerate(edges):
        zorder = 4 if i < 4 else 2
        ax.plot([corners_2d[e[0],0], corners_2d[e[1],0]], 
                [corners_2d[e[0],1], corners_2d[e[1],1]], f'{col}-', zorder=zorder)
    sz = corners_2d[1,0] - corners_2d[0,0]
    ax.add_patch(plt.Rectangle(corners_2d[0,:], sz, sz, 
                               fill=True, alpha=0.5, color='white', 
                               zorder=3))
    ax.axis('equal')
    ax.set_xlim(-W,W)
    ax.set_ylim(-H,H)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.grid(True)
    
    
def optimize(beta, translation, pp, proj, corners_2d, num_iter=1000):
    # Given a set of corners_2d, optimize beta and translation to fit the
    # corners_2d under projection function proj
    loss = nn.MSELoss()
    optimizer = optim.Adam([beta, translation], lr=0.1)
    for i in range(num_iter):
        optimizer.zero_grad()
        corners, z_axis = pp(beta, translation)
        corners_2d_pred = proj(corners)
        l = loss(corners_2d_pred, corners_2d)
        l.backward()
        optimizer.step()
        kp2d_error = torch.square(corners_2d_pred - corners_2d).sum(2).sqrt().mean(1)
        if l.item() < 1e-5:
            break
    # print(i, l.item())
    corners, z_axis = pp(beta, translation)
    corners_2d_pred = proj(corners)
    return l.item(), kp2d_error, z_axis, corners

def get_best_fit(corners_2d, pp, proj, num_iter=100, num_restarts=10):
    best_kp2d_error = np.inf
    for i in range(num_restarts):
        beta = torch.rand(1,3).float()*2-1
        beta[0,0] = torch.abs(beta[0,0])
        beta[0,1] = torch.abs(beta[0,1])
        beta.requires_grad_(True)
        translation = torch.rand(1,3).float()*2-1 
        translation[0,2] = torch.abs(translation[0,2])
        translation[0,1] = torch.abs(translation[0,1])
        translation.requires_grad_(True)
        error, kp2d_error, z_axis, corners = optimize(beta, translation, pp, proj, corners_2d, num_iter)
        if kp2d_error.item() < best_kp2d_error:
            best_kp2d_error = kp2d_error.item()
            best_beta = beta.clone().detach()
            best_translation = translation.clone().detach()
    return best_kp2d_error, best_beta, best_translation
