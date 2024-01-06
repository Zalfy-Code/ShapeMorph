import torch
from torch.utils import data

import numpy as np


import h5py
import os

category_ids = {

    '02691156': 0, # airphone
    '02828884': 1, # bench
    '02933112': 2, # cabinet
    '02958343': 3, # car
    '03001627': 4, # chair
    '03211117': 5, # tv
    '03636649': 6, # lamp
    '03691459': 7, # speaker
    '04090263': 8, # rifle
    '04256520': 9, # sofa
    '04379243': 10,# table
    '04401088': 11,# telephone
    '04530566': 12,# vessel

}



class ShapeNet(data.Dataset):
    def __init__(self, dataset_folder, split, categories=None, transform=None, sampling=True, num_samples=4096,
                 return_surface=True, surface_sampling=True, context_N=1024, pc_size=2048, grid_dim=64):

        self.pc_size = pc_size

        self.transform = transform
        self.num_samples = num_samples
        self.sampling = sampling
        self.split = split

        self.dataset_folder = dataset_folder
        self.return_surface = return_surface
        self.surface_sampling = surface_sampling

        self.dataset_folder = ''
        self.point_folder = ''
        self.mesh_folder = ''

        # self.partial_selector = VirtualScanSelector(context_N=context_N)

        if categories is None:
            categories = os.listdir(self.dataset_folder)
            categories = [c for c in categories if
                          os.path.isdir(os.path.join(self.dataset_folder, c)) and c.startswith('0')]
        categories.sort()
        print(categories)

        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(self.dataset_folder, c)
            assert os.path.isdir(subpath)

            split_file = os.path.join(subpath, split + '.lst')
            with open(split_file, 'r') as f:
                models_c = f.read().split('\n')

            self.models += [
                {'category': c, 'model': m.replace('.npz', '')}
                for m in models_c
            ]

        self.length = len(self.models)

    def __getitem__(self, idx):

        index = idx % self.length
        o_ind = index

        category = self.models[idx]['category']
        model = self.models[idx]['model']

        point_path = os.path.join(self.point_folder, category, model + '.npz')

        try:
            with np.load(point_path) as data:
                vol_points = data['vol_points']
                vol_label = data['vol_label']
                near_points = data['near_points']
                near_label = data['near_label']
        except Exception as e:
            print(e)
            print(point_path)

        with open(point_path.replace('.npz', '.npy'), 'rb') as f:
            scale = np.load(f).item()


        if self.return_surface:
            pc_path = os.path.join(self.dataset_folder, category, '4_pointcloud', model + '.npz')

            with np.load(pc_path) as data:
                surface = data['points'].astype(np.float32)
                point_cloud = surface * scale

            if self.surface_sampling:
                ind = np.random.default_rng().choice(point_cloud.shape[0], self.pc_size, replace=False)
                surface = point_cloud[ind]
            surface = torch.from_numpy(surface).float()
            Xct = surface

        if self.sampling:
            ind = np.random.default_rng().choice(vol_points.shape[0], self.num_samples, replace=False)
            vol_points = vol_points[ind]
            vol_label = vol_label[ind]

            ind = np.random.default_rng().choice(near_points.shape[0], self.num_samples, replace=False)
            near_points = near_points[ind]
            near_label = near_label[ind]

        vol_points = torch.from_numpy(vol_points)
        vol_label = torch.from_numpy(vol_label).float()

        if self.split == 'train':
            near_points = torch.from_numpy(near_points)
            near_label = torch.from_numpy(near_label).float()

            points = torch.cat([vol_points, near_points], dim=0)
            labels = torch.cat([vol_label, near_label], dim=0)
        else:
            points = vol_points
            labels = vol_label


        if self.transform:
            surface = self.transform(surface)

        if self.return_surface:
            return points, labels, surface, Xct
        else:
            return points, labels


    def __len__(self):
        return self.length


if __name__ == '__main__':
    pass