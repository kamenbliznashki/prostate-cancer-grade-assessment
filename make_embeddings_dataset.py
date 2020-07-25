import os
import argparse
import random
import glob
import json
import pprint

from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter

from train import load_df, remove_no_extractions_from_targets, make_dataloader, make_model
from dataset import EmbedderDataset


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cuda', type=int, help='Which GPU to use.')
# data
parser.add_argument('--tiles_dir', type=str, default='./data/train_images/tiles_level_1_overlap_0/')
parser.add_argument('--stats_dir', type=str, default='./data/train_images/tiles_level_1_overlap_0/stats/')
parser.add_argument('--df_path', default='./data/train.csv')
parser.add_argument('--restore_ckpt', type=str)
# params from training
parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--valid_batch_size', type=int, default=512)
parser.add_argument('--n_workers', type=int, default=8)
parser.add_argument('--bias_model', action='store_true')
# logging
parser.add_argument('--step', type=int, default=0, help='Current step (used for restoring models).')
parser.add_argument('--epoch', type=int, default=0, help='Current epoch (used for restoring models).')
parser.add_argument('--best_score', type=float, default=float('-inf'), help='Best validation scores.')
# model
parser.add_argument('--model', default='resnet34')
parser.add_argument('--n_classes', type=int, default=6, help='ISUP grading in {0,...,5}')
parser.add_argument('--pretrained', default=False, action='store_true', help='Use ImageNet pretrained model.')
parser.add_argument('--normalize_by_site', action='store_true')
# output
parser.add_argument('--save_dir', default='.')
parser.add_argument('--save_comments', type=str, nargs='*')


def load_config(args):
    config_path = os.path.join(os.path.dirname(args.restore_ckpt), 'config.json')
    if not os.path.exists(config_path):
        print('no config found at: ', config_path)
        return

    with open(config_path, 'r') as f:
        config = json.load(f)

    # update args
    print('updating args from config: ', config_path)
    args.normalize_by_site = config['normalize_by_site']

def make_dataset(df, args):
    return EmbedderDataset(args.tiles_dir, df)

@torch.no_grad()
def embed_dataset(model, dl, args):
    model.eval()
    idxs, embeddings, logits = [], [], []
    for idx, x in tqdm(dl):
        x = x.to(args.device)
        embeddings_, logits_ = model(x)

        idxs.extend(idx)
        embeddings.append(embeddings_.cpu())
        logits.append(logits_.cpu())

    embeddings = torch.cat(embeddings)
    logits = torch.cat(logits)

    print('saving embeddings to: ', args.save_path)
    torch.save({'tile_ids': idxs,
                'embeddings': embeddings,
                'logits': logits}, args.save_path)
    print(f'done. labels length {len(idxs)}, embeddings shape: {embeddings.shape}, logits shape: {logits.shape}')

def main(args):
    if args.model == 'debug':
        args.n_workers = 2
        args.train_batch_size = 32
        args.valid_batch_size = 32

    # setup device
    args.device = 'cuda:{}'.format(args.cuda) if args.cuda is not None and torch.cuda.is_available() else 'cpu'

    # seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if 'cuda' in args.device: torch.cuda.manual_seed(args.seed)

    # data
    targets_df = load_df(args)
    targets_df = remove_no_extractions_from_targets(targets_df, args)
    ds = make_dataset(targets_df, args)
    # NOTE -- no transforms here;
    # TODO -- can alternatively embed the dataset looping multiple epochs with random transforms
    dl = make_dataloader(ds, 'valid', args.valid_batch_size, shuffle=False, args=args)

    # model
    args.output_dim = 1  # not relevant here since only embedding
    model = make_model(args.model, args).to(args.device)

    # restore model and load config
    if args.restore_ckpt:
        ckpt = torch.load(args.restore_ckpt, map_location=args.device)
        model.load_state_dict(ckpt['model_state'])
        args.step = ckpt['step']
        args.epoch = ckpt['epoch']
        args.best_score = ckpt['best_score']
        print('restored checkpoint at epoch {}; best score {:.3f}'.format(args.epoch, args.best_score))

        load_config(args)

    # saving
    save_name = 'embeddings_{}_score_{:.3f}_step_{}'.format(args.model, args.best_score, args.step)
    if args.save_comments is not None: save_name += '_' + '_'.join(args.save_comments)
    args.save_path = os.path.join(args.save_dir, save_name + '.pt')

    embed_dataset(model, dl, args)



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
