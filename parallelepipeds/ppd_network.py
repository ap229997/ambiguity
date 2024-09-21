import os
import numpy as np
import torch
from torch import nn, optim
from ppd import Parallelopiped, Project
from torch.utils.data import Dataset
from absl import app, flags
torch.set_num_threads(4)

from absl import flags, app
FLAGS = flags.FLAGS
flags.DEFINE_float('lr', 1e-4, '')
flags.DEFINE_integer('num_iters', 5000, '')
flags.DEFINE_string('version', 'baseline', '')
flags.DEFINE_string('output_type', 'relative', '')
flags.DEFINE_integer('focal_length', 192, '')
flags.DEFINE_integer('fov', 35, 'degrees')

class PPDDataset():
    def __init__(self, proj, pp, device, batch_size=32, train=True, output_type='relative', fov=35):
        self.pp = pp
        self.proj = proj
        self.train = train
        self.rng = np.random.RandomState(0) if self.train else np.random.RandomState(12345)
        self.batch_size = batch_size
        self.device = device
        self.output_type = output_type
        self.fov = fov

    def __len__(self):
        if self.train:
            return 1024
        else:
            return 128

    def __getitem__(self, idx):
        # Logic to generate the random data
        beta_np = np.zeros((self.batch_size, 3), dtype=np.float32)
        beta_np[:,0] = self.rng.uniform(0+1e-4, 1.2, size=(self.batch_size,))
        beta_np[:,2] = self.rng.uniform(0, np.pi/2-1e-4, size=(self.batch_size,))
        beta_np[:,1] = self.rng.uniform(-np.pi, np.pi, size=(self.batch_size,))
        beta = torch.from_numpy(beta_np).float().to(self.device)
        
        xy_range = np.tan(self.fov * np.pi/180.)
        translation_np = self.rng.uniform(-xy_range, xy_range, size=(self.batch_size,3))
        translation_np[:,-1] = 1
        translation = torch.from_numpy(translation_np).float().to(self.device)
        
        corners_3d, z_axis = self.pp(beta, translation)
        corners_2d = self.proj(corners_3d).detach()
        if self.output_type == 'relative':
            # Root relative 3D Coordinates
            corners_3d = corners_3d - corners_3d.mean(1, keepdims=True)
        return corners_2d, corners_3d
    
def main(_):
    # Set up a network
    device = torch.device('cuda:0')
    
    pp = Parallelopiped(xy_size=0.2)
    proj = Project(f=FLAGS.focal_length)
    pp = pp.to(device) 
    
    batch_size = 32
    dataset = PPDDataset(proj, pp, device, batch_size, train=True, output_type=FLAGS.output_type, fov=FLAGS.fov)
    
    net = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), 
                        nn.Linear(64, 128), nn.ReLU(),
                        nn.Linear(128, 128), nn.ReLU(),
                        nn.Linear(128, 128), nn.ReLU(),
                        nn.Linear(128, 128), nn.ReLU(),
                        nn.Linear(128, 24))
    net = net.to(device)
    optimizer = optim.AdamW(net.parameters(), lr=FLAGS.lr)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    train_metric = []
    train_metric_save, val_metric_save = [], []
    for i in range(FLAGS.num_iters):
        x, y = dataset[i]
        x = x / 10000
        x_baseline = x - x.mean(1, keepdims=True)
        # Input to network is x or x_baseline. We think when input is x the
        # network will do better.
        # out = net(x_baseline.flatten(1))
        if FLAGS.version == 'baseline':
            out = net(x_baseline.flatten(1))
        else:
            out = net(x.flatten(1))

        # out = net(x.flatten(1))
        l = loss_fn(out, y.flatten(1))
        metric = out.reshape_as(y) - y
        # Mean Euclidean Distance
        metric = torch.sqrt(torch.square(metric).sum(2)).mean(1).mean(0)
        train_metric.append(metric.item())
        l.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 100 == 0:
            # print(i, l.item(), metric.item())
            print('iters:', i)
            print('train:', np.mean(train_metric))
            train_metric_save.append([i, np.mean(train_metric)])
            train_metric = []
            # validation
            with torch.no_grad():
                dataset_val = PPDDataset(proj, pp, device, batch_size, train=False, output_type=FLAGS.output_type, fov=FLAGS.fov)
                val_metric = []
                for j in range(int(len(dataset_val)/batch_size)):
                    x, y = dataset_val[j]
                    x = x / 10000
                    x_baseline = x - x.mean(1, keepdims=True)
                    if FLAGS.version == 'baseline':
                        out = net(x_baseline.flatten(1))
                    else:
                        out = net(x.flatten(1))
                    metric = out.reshape_as(y) - y
                    metric = torch.sqrt(torch.square(metric).sum(2)).mean(1).mean(0)
                    val_metric.append(metric.item())
            # print(i, 'val', metric.item())
            print('val:', np.mean(val_metric))
            print()
            val_metric_save.append([i, np.mean(val_metric)])

        # drop LR
        if i == 2500:
            for g in optimizer.param_groups:
                g['lr'] = FLAGS.lr / 10

    # save
    save_dir = './output'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(f'{save_dir}/{FLAGS.version}_train_metric_lr{FLAGS.lr}_{FLAGS.output_type}_f{FLAGS.focal_length}.npy', train_metric_save)
    np.save(f'{save_dir}/{FLAGS.version}_val_metric_lr{FLAGS.lr}_{FLAGS.output_type}_f{FLAGS.focal_length}.npy', val_metric_save)


if __name__ == '__main__':
    app.run(main)
