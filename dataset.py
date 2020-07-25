"""
References:
https://github.com/MSKCC-Computational-Pathology/MIL-nature-medicine-2019/blob/master/MIL_train.py

"""
import os
import glob
import random
from collections import Counter

from PIL import Image
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class MILdataset(Dataset):
    def __init__(self, tiles_dir, df):
        self.tiles_dir = tiles_dir
        self.df = df
        self.mode = None
        self.transform = None
        self.normalize_by_site = False

        self.all_tiles = sorted(glob.glob(tiles_dir + '*.png'))
        self.init(self.df.index)

        try:
            # number of tiles matches the number of slide indices
            assert len(self.tiles) == len(self.slide_idxs), 'Number of tiles does not match number of slide_idxs.'
            a, b, c = np.unique(self.slide_idxs, return_index=True, return_counts=True)
            # check slide idxs are repeated correct number of times
            assert np.all(b[1:] == np.cumsum(list(self.counter.values()))[:-1]), 'Mismatch in number of times slide_idxs are repeated.'
            # check slide idxs unique counts match the tiles counter
            assert np.all(c == list(self.counter.values()))
            # check slide idxs unique counts matches the size of the df
            assert len(c) == len(self.df)
            if 'isup_grade' in self.df.columns:
                # check number of unique slide idxs match number of targets
                assert len(a) == len(self.df), 'Mismatch between number of unique slide ids and number of targets provided. Check --debug flag if running locally.'
                # check idx of each unique slide id (b) maps to tiles with distinct image ids
                assert np.all(self.df.index.intersection([os.path.basename(self.tiles[i]).split('_')[0] for i in b]) == self.df.index)
        except AssertionError:
            print('Dataset construction mismatch.')

    def init(self, img_ids):
        self.tiles = [t for t in self.all_tiles if os.path.basename(t).split('_')[0] in img_ids]
        self.counter = Counter([os.path.basename(x).split('_')[0] for x in self.tiles])
        self.slide_idxs = np.array([i for i, (idx, count) in enumerate(self.counter.items()) for _ in range(count)])

        self.slide2img = {i: img_id for i, (img_id, _) in enumerate(self.counter.items())}
        self.img2slide = {img_id: i for i, img_id in self.slide2img.items()}

    def set_mode(self, mode):
        self.mode = mode

    def make_train_data(self, idxs):
        self.train_data = [(self.tiles[idx], self.df.loc[self.slide2img[self.slide_idxs[idx]]].isup_grade) for idx in idxs]

    def resample(self, classes_to_resample):
        if False:
            print('pre resampling')
            print(f'  df len = {len(self.df)}, tiles len = {len(self.tiles)}')
            print('  class counts: ', self.df.isup_grade.value_counts(sort=False).values)

        # counts for classes not to be resampled
        counts = self.df[~self.df.isup_grade.isin(classes_to_resample)].isup_grade.value_counts()

        # resample dataframe
        resampled_df = self.df[~self.df.isup_grade.isin(classes_to_resample)]
        for i in classes_to_resample:
            sampled_class_idxs = np.random.choice(np.flatnonzero(self.df.isup_grade.values==i),
                                                  size=int(np.mean(counts)), replace=False)
            resampled_df = resampled_df.append(self.df.iloc[sampled_class_idxs])

        # update tiles, counter and slide idxs
        self.init(resampled_df.index)

        if False:
            print('post resampling')
            print(f'  df len = {len(resampled_df)}, tiles len = {len(self.tiles)}')
            print('  class counts: ', resampled_df.isup_grade.value_counts(sort=False).values)

    def process_image(self, img_path):
        img = Image.open(img_path)

        if self.normalize_by_site:
            img_id = os.path.basename(img_path).split('_')[0]
            site = self.df.loc[img_id].data_provider
            img = self.transform[site](img)
        else:
            img = self.transform(img)

        return img

    def __len__(self):
        if self.mode == 'eval':
            return len(self.tiles)
        elif self.mode == 'train':
            return len(self.train_data)

    def __getitem__(self, idx):
        if self.mode == 'eval':
            img_path = self.tiles[idx]
        elif self.mode == 'train':
            img_path, target = self.train_data[idx]

        img = self.process_image(img_path)

        if self.mode == 'eval':
            return img
        elif self.mode == 'train':
            return img, target


class MILdatasetTopk(MILdataset):
    def __init__(self, tiles_dir, df, k):
        super().__init__(tiles_dir, df)
        self.k = k

    def make_train_data(self, idxs):
        super().make_train_data(idxs)
        self.train_slides = np.array([self.slide_idxs[i] for i in idxs])

    def make_seq_data(self, idxs, groups):
        self.seq_tiles  = [(self.tiles[i], None) for i in idxs] # keep symmetric to train_data
        self.seq_slides = np.array([self.slide_idxs[i] for i in idxs])

    def __len__(self):
        if self.mode == 'eval':
            return len(self.tiles)
        elif self.mode == 'eval_seq':
            return len(self.df)
        elif self.mode == 'train':
            return len(self.df)

    def make_img_seq(self, tile_paths, slide_idxs, slide_idx):
        # use the indices in slide idxs to index tile paths
        idxs = np.where(slide_idxs == slide_idx)[0]

        # process every image in the sequence for this slide
        imgs = []
        for i in idxs:
            img_path, target = tile_paths[i]
            # NOTE -- this applies the transforms to each so can have different augmentations
            imgs.append(self.process_image(img_path))
        imgs = torch.stack(imgs)

        # pad if under k
        if imgs.shape[0] < self.k:
            imgs = F.pad(imgs, (0,0,0,0,0,0,0,self.k - imgs.shape[0]), 'constant', 0)

        return imgs, target

    def __getitem__(self, idx):
        if self.mode == 'eval':
            img_path = self.tiles[idx]
            img = self.process_image(img_path)
        elif self.mode == 'eval_seq':
            img, _ = self.make_img_seq(self.seq_tiles, self.seq_slides, idx)
        elif self.mode == 'train':
            img, target = self.make_img_seq(self.train_data, self.train_slides, idx)

        if self.mode == 'train':
            return img, target
        else:
            return img



class EmbedderDataset(MILdataset):
    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        img_path = self.tiles[idx]
        tile_id = os.path.basename(img_path)[:-4]  # exclude .png
        img_id = tile_id.split('_')[0]

        img = Image.open(img_path)

        if self.normalize_by_site:
            site = self.df.loc[img_id].data_provider
            img = self.transform[site](img)
        else:
            img = self.transform(img)

        return tile_id, img


if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)

    tiles_dir = './data/train_images/tiles/'
    df = pd.read_csv('./data/train.csv').set_index('image_id').sort_index()

    ds = MILdataset(tiles_dir, df)

    # test dataset length under train and eval
    if False:
        ds.set_mode('eval')
        print('Eval mode len = ', len(ds))

        from train import group_argtopk
        probs = np.random.rand(len(ds), 6)
        idxs = group_argtopk(probs, ds.slide_idxs)
        ds.make_train_data(idxs)

        ds.set_mode('train')
        print('Train mode len = ', len(ds))
        print('Label counts = ', np.histogram([y for x,y in ds.train_data], bins=6)[0])

        ds.shuffle_train_data()

    # test resampling
    if False:
        ds = MILdataset(tiles_dir, df, [0,1])

        ds.set_mode('eval')
        print('Eval mode len = ', len(ds))

        from train import group_argtopk
        probs = np.random.rand(len(ds), 6)
        idxs = group_argtopk(probs, ds.slide_idxs)
        ds.make_train_data(idxs)

        ds.set_mode('train')
        print('Train mode len = ', len(ds))
        print('Label counts = ', np.histogram([y for x,y in ds.train_data], bins=6)[0])

        ds.resample()

        print('Train mode len = ', len(ds))
        print('Label counts = ', np.histogram([y for x,y in ds.train_data], bins=6)[0])

    # test resampling on local data
    if False:
        # take df matching locally available data
        import os
        img_ids = np.unique([os.path.basename(x).split('_')[0] for x in ds.all_tiles])
        local_df = df.loc[img_ids]

        ds = MILdataset(tiles_dir, local_df)

        # no resampling -------------
        ds.set_mode('eval')
        print('Eval mode len = ', len(ds))

        from train import group_argtopk
        probs = np.random.rand(len(ds), 6)
        idxs = group_argtopk(probs, ds.slide_idxs)
        ds.make_train_data(idxs)

        ds.set_mode('train')
        print('Train mode len = ', len(ds))
        print('Label counts = ', np.histogram([y for x,y in ds.train_data], bins=6)[0])

        # resampling ----------------
        print('-------- resampling classes 0,1 --------')
        ds.resample([0,1])

        ds.set_mode('eval')
        print('Eval mode len = ', len(ds))

        probs = np.random.rand(len(ds), 6)
        idxs = group_argtopk(probs, ds.slide_idxs)
        ds.make_train_data(idxs)

        ds.set_mode('train')
        print('Train mode len = ', len(ds))
        print('Label counts = ', np.histogram([y for x,y in ds.train_data], bins=6)[0])

        # resampling ----------------
        print('-------- resampling class 5 --------')
        ds.resample([4])

        ds.set_mode('eval')
        print('Eval mode len = ', len(ds))

        probs = np.random.rand(len(ds), 6)
        idxs = group_argtopk(probs, ds.slide_idxs)
        ds.make_train_data(idxs)

        ds.set_mode('train')
        print('Train mode len = ', len(ds))
        print('Label counts = ', np.histogram([y for x,y in ds.train_data], bins=6)[0])



    # show image from dataset and matching image manually
    if False:
        import matplotlib.pyplot as plt
        from skimage.io import MultiImage
        plt.figure(figsize=(16,8))
        plt.subplot(1,2,1)
        plt.imshow(ds[0][0])
        plt.subplot(1,2,2)
        img_path = os.path.join('./data/train_images/', os.path.basename(ds.train_data[0][0]).split('_')[0] + '.tiff')
        img = MultiImage(img_path)
        plt.imshow(img[1])
        plt.show()


    # setup dataloader and save dataset batch sample
    if False:
        from torch.utils.data import DataLoader
        from torchvision.utils import save_image
        import torchvision.transforms as T

        # 0 background
        mean = np.array([0.67496395, 0.50394961, 0.63128368])
        std  = np.array([0.31837059, 0.29695359, 0.29912062])

        batch_size = 16
        transform = T.Compose([T.RandomRotation(30, fill=0),
                               T.ToTensor(),
                               T.Normalize(mean, std),
                               ])
        ds = MILdataset(tiles_dir, targets_path, transform)
        ds.set_mode('eval')
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

        batch = next(iter(dl))
        print('batch shape = ', batch.shape)
        save_image(batch, 'sample_batch.png')


    # test embeddings dataset in identity model
    if False:
        tiles = glob.glob(tiles_dir + '*.png')
        local_df = df.loc[[os.path.basename(x).split('_')[0] for x in tiles]]
        ds = EmbeddingsDataset('embeddings_debug_score_0.000_step_2.pt', local_df, 'train', method='identity')
        print(f'dataset length = {len(ds)}')

        item = ds[0]
        print(f'ds element -- embedding shape = {item[0].shape}, label = {item[1]}')

        # order in ds should match order in df =>
        # 1. label should be the same
        assert item[1] == local_df.iloc[0].isup_grade
        # 2. shape of returned embedding should match the number of existing tiles for this img id
        assert item[0].shape[0] == len([x for x in tiles if os.path.basename(x).split('_')[0] == local_df.iloc[0].name])

    # test embeddings dataset in mean mode
    if False:
        tiles = glob.glob(tiles_dir + '*.png')
        local_df = df.loc[[os.path.basename(x).split('_')[0] for x in tiles]]
        ds = EmbeddingsDataset('embeddings_debug_score_0.000_step_2.pt', local_df, 'train', method='mean')
        from torch.utils.data import DataLoader
        dl = DataLoader(ds, batch_size=5, shuffle=False)
        batch = next(iter(dl))
        print(f'dl batch element -- batch embeddings shape = {batch[0].shape}, batch labels shape = {batch[1].shape}')

    # test embeddings dataset in topk mode
    if True:
        tiles = glob.glob(tiles_dir + '*.png')
        local_df = df.loc[[os.path.basename(x).split('_')[0] for x in tiles]]
        ds = EmbeddingsDataset('embeddings_debug_score_0.000_step_2.pt', local_df, 'train', method='topk')

        tiles, label = ds[0]
        print(f'ds element -- selected tiles shape = {tiles.shape}, label = {label}')
        print(tiles)

        from torch.utils.data import DataLoader
        dl = DataLoader(ds, batch_size=5, shuffle=False)
        batch = next(iter(dl))
        print(f'dl batch element -- batch embeddings shape = {batch[0].shape}, batch labels shape = {batch[1].shape}')

