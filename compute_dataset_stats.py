import os
import glob
import argparse
import multiprocessing as mp

import numpy as np
import pandas as pd
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--img_id', type=str)
parser.add_argument('--tiles_dir', type=str, default='./data/train_images/tiles/')
parser.add_argument('--compute_by_site', action='store_true', help='Use csv file to compute image stats by site.')
parser.add_argument('--df_path', type=str, default='./data/train.csv')


def compute_img_stats(img_path):
    im = Image.open(img_path)
    img = np.asarray(im)

    x_tot = np.mean(img / 255, axis=(0,1))
    x2_tot = np.mean((img / 255)**2, axis=(0,1))

    im.close()

    return x_tot, x2_tot

def aggregate_stats(filepaths, save_prefix='img'):
    with mp.Pool() as p:
        out = p.map(compute_img_stats, filepaths)

    # compute mean and std of all processed tiles
    x_tot  = [a for a,b in out]
    x2_tot = [b for a,b in out]

    img_avg = np.mean(x_tot, 0)
    img_std = np.sqrt(np.mean(x2_tot, 0) - img_avg**2)
    print('img avg: ', img_avg)
    print('img std: ', img_std)

    np.savetxt(f'{save_prefix}img_avg.txt', img_avg)
    np.savetxt(f'{save_prefix}img_std.txt', img_std)


if __name__ == '__main__':
    args = parser.parse_args()

    if args.img_id:
        aggregate_stats(os.path.join(args.tiles_dir, img_id + '.png'))
    else:
        all_filepaths = glob.glob(args.tiles_dir + '*.png')

        if args.compute_by_site:
            df = pd.read_csv(args.df_path).set_index('image_id').sort_index()
            print('df len = ', len(df))

            for name, sub_df in df.groupby('data_provider'):
                filepaths = [f for f in all_filepaths if os.path.basename(f).split('_')[0] in sub_df.index]
                print(f'group: {name}, num patients = {len(sub_df)}, num tiles = {len(filepaths)}')
                aggregate_stats(filepaths, name + '_')

        else:
            aggregate_stats(all_filepaths)
