import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
import matplotlib

import seaborn as sns
sns.set_theme()

from ppd import Parallelopiped, Project, vis_corners, optimize, get_best_fit
out_dir = 'vis_ppd'
os.makedirs(out_dir, exist_ok=True)

def worker(corners0, corners0_2d, beta0, translation0, pp, proj):
    # Create metamers
    W, H = 112, 112
    offset = 32
    num_offsets = 30
    offsets_x = np.linspace(-W+offset, W-offset, num_offsets, dtype=np.float32)
    offsets_y = np.linspace(-H+offset, H-2*offset, num_offsets, dtype=np.float32)
    kp2d_errors = [0.]
    kp3d_errors = [0.]
    kp3d_relative_errors = [0.] 
    betas = [beta0]
    translations = [translation0]
    offsets = [0]

    for i, offset_x in enumerate(tqdm(offsets_x)):
        for j, offset_y in enumerate(offsets_y):
            offset2d = torch.tensor([offset_x, offset_y]).reshape((1,1,2))
            kp2d_error, beta, translation = get_best_fit(corners0_2d+offset2d, pp, proj, num_iter=100, num_restarts=10)
            corners, z_axis = pp(beta, translation)
            kp3d_relative_error = torch.sqrt(torch.square(corners - corners.mean(1, keepdims=True) - 
                                                          (corners0 - corners0.mean(1, keepdims=True))).sum(2)).mean(1)
            kp3d_error = torch.sqrt(torch.square(corners - corners0).sum(2)).mean(1)
            kp3d_relative_errors.append(kp3d_relative_error.item())
            kp3d_errors.append(kp3d_error.item())
            kp2d_errors.append(kp2d_error)
            betas.append(beta)
            translations.append(translation)
            offsets.append(np.sqrt(offset_x**2 + offset_y**2))
    offsets = np.array(offsets)
    kp2d_errors = np.array(kp2d_errors)
    kp3d_errors = np.array(kp3d_errors)
    kp3d_relative_errors = np.array(kp3d_relative_errors)

    return offsets, betas, translations, kp2d_errors, kp3d_errors, kp3d_relative_errors
    
def get_ind_to_plot(kp3d_errors, kp2d_errors, kp2d_errors_thresh):
    ind_to_consider = kp2d_errors < kp2d_errors_thresh
    _kp3d_errors = kp3d_errors+0
    _kp3d_errors[np.invert(ind_to_consider)] = -np.inf
    ind = np.argsort(_kp3d_errors)[::-1]
    ind = [0] + ind.tolist()
    ind = np.array(ind)
    return ind

def plot1(kp2d_errors, kp3d_errors, kp3d_relative_errors, offsets):
    # scatter plot err_2d_pix, err_3d_mm, large displacements
    thresh1 = 20
    thresh2 = 30


    cm = matplotlib.colormaps['coolwarm']
    xlim = [0, 1]
    ylim = [0, 1]
    xlabel = '2D Keypoint Error (pixels)'
    clabel = 'Distance between crops (pixels)'

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    sc1 = ax.scatter(kp2d_errors, kp3d_errors,
                    alpha=0.5, linewidths=0, c=offsets, 
                    cmap=cm, vmin=0, vmax=thresh2*2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('3D Keypoint Error (m)')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.colorbar(sc1, label=clabel)
    plt.savefig(f'{out_dir}/parallelepipeds-plot.pdf', bbox_inches='tight')

    ylim = [0, 0.1]
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    sc2 = ax.scatter(kp2d_errors, kp3d_relative_errors,
                     alpha=0.5, linewidths=0, c=offsets, 
                     cmap=cm, vmin=0, vmax=thresh2*2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('3D Keypoint Error (after centering) (m)')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    plt.colorbar(sc2, label=clabel)
    plt.savefig(f'{out_dir}/parallelepipeds-plot-centering.pdf', bbox_inches='tight')

    # ind = dist2d_kps <= 4
    # ind_to_plot = np.logical_and(dist2d_kps <= 4, IDX)

def plot3(kp2d_errors, kp3d_errors, kp3d_relative_errors, offsets, beta0,
          translation0, betas, translations, pp, proj):
    big_image = False 
    extra_fig = 2 if big_image else 0
    N = min(5, len(kp3d_errors))
    figsize = 4./5
    fontsize = 24

    fig = plt.figure(figsize=((N+extra_fig)*figsize*5,2*figsize*5))
    gs = GridSpec(2, N+extra_fig, left=0, right=1, top=0.99, bottom=0.01, wspace=0.01, hspace=0.01)
    
    corners0, z_axis0 = pp(beta0, translation0)
    corners0_2d = proj(corners0).detach()
    
    if big_image:
        ax = fig.add_subplot(gs[:2,:2])
        vis_corners(ax, corners0_2d[0])

    zero_translation = torch.tensor([0., 0., 0.5]).reshape(1,3)
    ind = get_ind_to_plot(kp3d_relative_errors, kp2d_errors, 1)

    for i in range(N):    
        ax = fig.add_subplot(gs[0,i+extra_fig])
        corners, z_axis = pp(betas[ind[i]], zero_translation)
        corners_2d = proj(corners).detach()
        vis_corners(ax, corners_2d[0])
        if i == 0:
            title_str = 'True 3D for image'
        else:
            title_str = f'Rel. 3D KP Err: {kp3d_relative_errors[ind[i]].item():0.2f}m\n Abs. 3D KP Err: {kp3d_errors[ind[i]].item():0.2f}m'
        ax.set_title(title_str, fontdict={'fontsize': fontsize}, y=1, pad=-64)

        ax = fig.add_subplot(gs[1,i+extra_fig])
        corners, z_axis = pp(betas[ind[i]], translations[ind[i]])
        corners_2d = proj(corners).detach()
        vis_corners(ax, corners_2d[0])
        ax.set_title(f'2D KP Error: {kp2d_errors[ind[i]]:0.2f}px \n2D Shift: {offsets[ind[i]]:0.2f}px', 
                     fontdict={'fontsize': fontsize}, y=1, pad=-256)
        # ax.plot(joints_r_pnp_delta[alias_id[i], :, 0], joints_r_pnp_delta[alias_id[i], :, 1]-32, 'r.', ms=1)
        # ax.axis(False)
        # ax.set_xticks([])
        # ax.set_yticks([])
    name = '-combined'
    plt.savefig(f'{out_dir}/parallelepipeds-vis{name}.pdf', bbox_inches='tight')



def plot2(kp2d_errors, kp3d_errors, kp3d_relative_errors, offsets, beta0,
          translation0, betas, translations, pp, proj):
    big_image = False 
    extra_fig = 2 if big_image else 0
    for errors_3d, name in zip([kp3d_errors, kp3d_relative_errors], ['', '-centering']):
        N = min(8, len(errors_3d))
        figsize = 4./5
        fontsize = 24

        fig = plt.figure(figsize=((N+extra_fig)*figsize*5,2*figsize*5))
        gs = GridSpec(2, N+extra_fig, left=0, right=1, top=0.99, bottom=0.01, wspace=0.01, hspace=0.01)
        
        corners0, z_axis0 = pp(beta0, translation0)
        corners0_2d = proj(corners0).detach()
        
        if big_image:
            ax = fig.add_subplot(gs[:2,:2])
            vis_corners(ax, corners0_2d[0])

        zero_translation = torch.tensor([0., 0., 0.5]).reshape(1,3)
        ind = get_ind_to_plot(errors_3d, kp2d_errors, 1)

        for i in range(N):    
            ax = fig.add_subplot(gs[0,i+extra_fig])
            corners, z_axis = pp(betas[ind[i]], zero_translation)
            corners_2d = proj(corners).detach()
            vis_corners(ax, corners_2d[0])
            if i == 0:
                title_str = 'True 3D for image'
            else:
                title_str = f'3D KP Error: {errors_3d[ind[i]].item():0.2f}m'
            ax.set_title(title_str, fontdict={'fontsize': fontsize}, y=1, pad=-32)

            ax = fig.add_subplot(gs[1,i+extra_fig])
            corners, z_axis = pp(betas[ind[i]], translations[ind[i]])
            corners_2d = proj(corners).detach()
            vis_corners(ax, corners_2d[0])
            ax.set_title(f'2D KP Error: {kp2d_errors[ind[i]]:0.2f}px \n2D Shift: {offsets[ind[i]]:0.2f}px', 
                         fontdict={'fontsize': fontsize}, y=1, pad=-256)
            # ax.plot(joints_r_pnp_delta[alias_id[i], :, 0], joints_r_pnp_delta[alias_id[i], :, 1]-32, 'r.', ms=1)
            # ax.axis(False)
            # ax.set_xticks([])
            # ax.set_yticks([])
        plt.savefig(f'{out_dir}/parallelepipeds-vis{name}.pdf', bbox_inches='tight')


def main():
    pp = Parallelopiped(xy_size=0.2)
    proj = Project(f=192)

    phi = 0.
    beta0 = torch.from_numpy(np.array([[0.20, 0, phi]])).float()
    translation0 = torch.from_numpy(np.array([[0.,0.,1.]])).float()
    corners0, z_axis0 = pp(beta0, translation0)
    corners0_2d = proj(corners0).detach()

    offsets, betas, translations, kp2d_errors, kp3d_errors, kp3d_relative_errors = worker(
        corners0, corners0_2d, beta0, translation0, pp, proj)
    plot1(kp2d_errors, kp3d_errors, kp3d_relative_errors, offsets)
    plot2(kp2d_errors, kp3d_errors, kp3d_relative_errors, offsets, beta0, translation0, betas, translations, pp, proj)
    plot3(kp2d_errors, kp3d_errors, kp3d_relative_errors, offsets, beta0, translation0, betas, translations, pp, proj)

if __name__ == '__main__':
    main()
