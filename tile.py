import os
import glob
import json
import argparse
import multiprocessing as mp
from functools import partial
from itertools import product

import numpy as np
import pandas as pd
from PIL import Image
from skimage.io import MultiImage
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.filters import threshold_otsu
from skimage.morphology import binary_dilation, square
from skimage.util import img_as_ubyte

parser = argparse.ArgumentParser()
parser.add_argument('--img_id', type=str)
# paths
parser.add_argument('--tiff_dir', type=str, default='./data/train_images/')
parser.add_argument('--tiles_dir', type=str, default='./data/train_images/tiles/')
parser.add_argument('--df_path', type=str, default='./data/train.csv')
# settings
parser.add_argument('--overlap', type=float, default=0.5, help='Percent overlap among tiles.')
parser.add_argument('--auto_adjust_overlap', type=eval, default=False, help='Automatically lower overlap if using larger images')
parser.add_argument('--tile_size', type=float, default=224, help='Tile size.')

def make_mask(img):
    gray_img = rgb2gray(img)

    # convert back to [0, 255]
    gray_img = img_as_ubyte(gray_img)

    # Otsu's thresholding
    thresh = threshold_otsu(gray_img.astype(int))
    mask = gray_img < thresh

    # dilation on the mask (returns bool)
    mask = binary_dilation(mask, square(5))

    return mask

def resize_mask(mask, target_shape):
    mask = resize(mask, target_shape, order=0)  # resize return float
    return mask.astype(bool)

def remove_img_background(img, mask, old_val=255, new_val=0):
    # replace background where mask is False and all channels equal old_val
    img[~mask & np.all(img == old_val, axis=2), :] = new_val
    return img

def make_tiles(img, mask, overlap, size):
    H, W, _ = img.shape

    stride = int(size * (1 - overlap))

    hpad = int((np.ceil(H / stride) - 1) * stride + size - H)
    wpad = int((np.ceil(W / stride) - 1) * stride + size - W)

    img_padded  = np.pad(img,  [[hpad // 2, hpad - hpad // 2], [wpad // 2, wpad - wpad // 2], [0,0]], constant_values=0)
    mask_padded = np.pad(mask, [[hpad // 2, hpad - hpad // 2], [wpad // 2, wpad - wpad // 2]],        constant_values=0)

    vcoords = [h for h in range(0,H,stride)]
    hcoords = [w for w in range(0,W,stride)]

    img_tiled, mask_tiled = [], []
    for h, w in product(vcoords, hcoords):
        xslice = slice(h, h+size)
        yslice = slice(w, w+size)
        img_tiled.append(img_padded[xslice, yslice])
        mask_tiled.append(mask_padded[xslice, yslice])

    img_tiled = np.stack(img_tiled)
    mask_tiled = np.stack(mask_tiled)

    return img_tiled, mask_tiled

def select_tiles(img_tiled, mask_tiled, thresh=0.5):
    # compute mean mask area for each tile
    area = np.mean(mask_tiled, axis=(1,2))

    # select only tiles with mean mask values greater than threshold
    top_area = area[area > thresh]

    # order subset descending
    order = np.argsort(-top_area)

    img_tiled_selected = img_tiled[area > thresh][order]

    return img_tiled_selected


def save_tiles(img_tiled, filename, target_dir):
    x_tot, x2_tot = [], []
    for i, img_tile in enumerate(img_tiled):
        tile_path = os.path.join(target_dir, filename + '_{:03d}.png'.format(i))

        with Image.fromarray(img_tile) as img:
            img.save(tile_path)

        x_tot.append(np.mean(img_tile / 255, axis=(0,1)))
        x2_tot.append(np.mean((img_tile / 255)**2, axis=(0,1)))

    return x_tot, x2_tot

def process_image(tiff_path, target_dir, overlap, auto_adjust_overlap, tile_size):
    # outputs
    img_id  = os.path.basename(tiff_path).split('.')[0]
    x_tot   = []
    x2_tot  = []
    success = True

    # processing steps:
    # 1. load tiff img
    # 2. make mask using small image
    # 3. extract tiles:
    #    a. select medium or large img to extract from
    #    b. resize mask to target img
    #    c. remove background from image
    #    d. make tiles
    #    e. select tiles where img not all white and mask covers % of image

    if success:
        try:
            tiff_img = MultiImage(tiff_path)
        except:
            print('error loading tiff image: ', tiff_path)
            success = False

    if success:
        try:
            mask = make_mask(tiff_img[2])  # make mask from small image
        except:
            print('error processing mask for: ', tiff_path)
            success = False

    if success:
        try:
            # if medium mask (extract from small, resize to medium) is covered by < 4 tiles,
            # use large image to resize mask and extract tiles from
            #   note -- medium mask covered by size x size tiles is equivalent to
            #           small mask covered by size/4 x size/4 tiles since each tiff img is at 4x higher resolution
            if mask.sum() / (tile_size * tile_size / 16) < 4:
                img = tiff_img[0]
                if auto_adjust_overlap:
                    overlap = 0.  # no overlap at highest resolution
            else:
                img = tiff_img[1]

            mask = resize_mask(mask, img.shape[:2])                             # resize mask to medium image shape
            img = remove_img_background(img, mask)                              # make background 0 pixel instead of 255
            img_tiled, mask_tiled = make_tiles(img, mask, overlap, tile_size)   # tile the medium sized img
            img_tiled = select_tiles(img_tiled, mask_tiled)                     # select tiles

        except:
            print('error making tiles for: ', tiff_path)
            success = False

    if success:
        try:
            x_tot, x2_tot = save_tiles(img_tiled, os.path.basename(tiff_path)[:-5], target_dir)
        except:
            print('error saving tiles for: ', tiff_path)
            success = False

    if success:
        if img_tiled.shape[0] == 0:
            print('no tiles extracted from: ', tiff_path)
            success = False

    return img_id, x_tot, x2_tot, success


if __name__ == '__main__':
    args = parser.parse_args()

    # make dirs for tiles and dataset stats within tiles
    args.stats_dir = os.path.join(args.tiles_dir, 'stats')
    os.makedirs(args.stats_dir, exist_ok=True)

    if args.img_id:
        process_image(os.path.join(args.tiff_dir, args.img_id + '.tiff'), args.tiles_dir)
    else:
        if os.path.exists(os.path.join(args.stats_dir, 'img_avg.txt')):
            raise RuntimeError('Files for img avg/std already exist.')

        df = pd.read_csv(args.df_path).set_index('image_id').sort_index()
        print('df len = ', len(df))

        # process everything in the tiff_dir
        with mp.Pool() as p:
            # process images
            #   out is list of tuples (img_id, x_tot, x2_tot, success)
            #   where img_id is str, x_tot and x2_tot are lists of arrays of size 3, success is bool
            out = p.map(partial(process_image, target_dir=args.tiles_dir, overlap=args.overlap,
                                auto_adjust_overlap=args.auto_adjust_overlap, tile_size=args.tile_size),
                        glob.glob(args.tiff_dir + '*.tiff'))

        # process output
        #   no extractions
        img_ids_with_no_tiles = [a for a,_,_,d in out if d is False]
        #   compute mean and std of all processed tiles
        x_tot  = [x for _,b,c,_ in out for x in b]
        x2_tot = [x for _,b,c,_ in out for x in c]
        img_avg = np.mean(x_tot, 0)
        img_std = np.sqrt(np.mean(x2_tot, 0) - img_avg**2)

        # save output
        #   save settings
        with open(os.path.join(args.stats_dir, 'config.json'), 'w') as f:
            json.dump(args.__dict__, f)

        #   save no extractions
        with open(os.path.join(args.stats_dir, 'img_ids_with_no_tiles.txt'), 'w') as f:
            f.write('\n'.join(img_ids_with_no_tiles))
        #   save mean and std
        np.savetxt(os.path.join(args.stats_dir, 'img_avg.txt'), img_avg)
        np.savetxt(os.path.join(args.stats_dir, 'img_std.txt'), img_std)

        print('aggregate stats:')
        print('img avg: ', img_avg)
        print('img std: ', img_std)

        # stats by site
        for name, sub_df in df.groupby('data_provider'):
            x_tot_site  = [x for a,b,c,_ in out for x in b if a in sub_df.index]
            x2_tot_site = [x for a,b,c,_ in out for x in c if a in sub_df.index]

            img_avg = np.mean(x_tot_site, 0)
            img_std = np.sqrt(np.mean(x2_tot_site, 0) - img_avg**2)

            np.savetxt(os.path.join(args.stats_dir, f'{name}_img_avg.txt'), img_avg)
            np.savetxt(os.path.join(args.stats_dir, f'{name}_img_std.txt'), img_std)

            print(f'group: {name}, num patients = {len(sub_df)}, num tiles = {len(x_tot_site)}')
            print('  img avg: ', img_avg)
            print('  img std: ', img_std)

