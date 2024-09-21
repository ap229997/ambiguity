import pickle
import torch
from ppd import Parallelopiped, Project
from ppd_network import PPDDataset

def generate(focal_length=192, fov=35, output_type='absolute', train=True, num_iters=5000):
    device = torch.device('cuda:0')
    pp = Parallelopiped(xy_size=0.2)
    proj = Project(f=focal_length)
    pp = pp.to(device) 
    
    batch_size = 32
    dataset = PPDDataset(proj, pp, device, batch_size, train=train, output_type=output_type, fov=fov)

    all_x, all_y = [], []
    for i in range(num_iters):
        x, y = dataset[i]
        all_x.append(x)
        all_y.append(y)

    all_x = torch.stack(all_x, dim=0)
    all_y = torch.stack(all_y, dim=0)

    return all_x.cpu().numpy(), all_y.cpu().numpy()

def main():
    rel_train_x, rel_train_y = generate(output_type='relative', train=True)
    rel_val_x, rel_val_y = generate(output_type='relative', train=False)
    abs_train_x, abs_train_y = generate(output_type='absolute', train=True)
    abs_val_x, abs_val_y = generate(output_type='absolute', train=False)
    
    data = {'relative': {'train': {'x': rel_train_x, 'y': rel_train_y}, 'val': {'x': rel_val_x, 'y': rel_val_y}},
            'absolute': {'train': {'x': abs_train_x, 'y': abs_train_y}, 'val': {'x': abs_val_x, 'y': abs_val_y}}}

    with open('ppd_dataset.pkl', 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    main()