import os
import argparse
import random
import glob
import json
import pprint
import subprocess
from functools import partial

from PIL import ImageFilter, Image
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter

from dataset import MILdataset, MILdatasetTopk
from models import make_model, Encoder

parser = argparse.ArgumentParser()
# actions
parser.add_argument('--train', action='store_true')
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--subset', type=int, help='Subset only this many slide examples from the data.')
#
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cuda', type=int, help='Which GPU to use.')
# data
parser.add_argument('--tiles_dir', type=str, default='./data/train_images/tiles_level_1_overlap_0/')
parser.add_argument('--stats_dir', type=str, default='./data/train_images/tiles_level_1_overlap_0/stats/')
parser.add_argument('--df_path', type=str, default='./data/train.csv')
parser.add_argument('--restore_ckpt', type=str)
parser.add_argument('--restore_encoder_ckpt', type=str)
# training
parser.add_argument('--k', type=int, default=1, help='Top k tiles assumed to be of the same class as the slide.')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--valid_batch_size', type=int, default=512)
parser.add_argument('--n_epochs', type=int, default=1)
parser.add_argument('--n_workers', type=int, default=8)
parser.add_argument('--loss', choices=['ce', 'bce', 'masked_bce', 'mse'])
parser.add_argument('--loss_weights', type=float, nargs=6)
parser.add_argument('--bias_model', action='store_true')
parser.add_argument('--normalize_by_site', action='store_true', help='Mean and std normalization by provider site.')
parser.add_argument('--resample_classes', type=int, nargs='*')
parser.add_argument('--augment', action='store_true', help='Data augmentation during training.')
# logging
parser.add_argument('--eval_interval', type=int, default=1, help='How often to evaluate model on validation set.')
parser.add_argument('--save_interval', type=int, default=1, help='How often to save model.')
parser.add_argument('--step', type=int, default=0, help='Current step (used for restoring models).')
parser.add_argument('--epoch', type=int, default=0, help='Current epoch (used for restoring models).')
parser.add_argument('--best_score', type=float, default=float('-inf'), help='Best validation scores updated during training.')
parser.add_argument('--commit', type=str, default=subprocess.check_output(['git', 'describe', '--always']).strip().decode(), help='Store the current git commit from which the script is executed.')
# model
parser.add_argument('--input', choices=['tiles', 'bags'], default='tiles')
parser.add_argument('--model', type=str, default='resnet34')
parser.add_argument('--encoder', type=str, help='If training with --input=bags, e.g. --encoder=resnet34 and --model=mil-mean)')
parser.add_argument('--n_classes', type=int, default=6, help='ISUP grading in {0,...,5}')
parser.add_argument('--pretrained', default=False, action='store_true', help='Use ImageNet pretrained model.')
parser.add_argument('--clf_thresh', type=float, default=0.0, help='Threshold over which to consider class positive.')


# --------------------
# Checkpointing utils
# --------------------

def save_ckpt(model, optimizer, is_best, log_dir, args):
    torch.save({'best_score': args.best_score,
                'epoch': args.epoch,
                'step': args.step,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict()},
               os.path.join(log_dir, 'checkpoint_best.pt' if is_best else 'checkpoint.pt'))

def save_json(content, path):
    if os.path.exists(path):
        print(f'Config not saved -- config.json already exists at: {path}')

    with open(path, 'w') as f:
        json.dump(content, f)

def load_ckpt(ckpt_path, model, optimizer, args, update_args):
    ckpt = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(ckpt['model_state'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optim_state'])
    epoch, best_score = ckpt['epoch'], ckpt['best_score']
    if update_args:
        args.step, args.epoch, args.best_score = ckpt['step'], ckpt['epoch'], ckpt['best_score']
    print('restored checkpoint (epoch {}, best score {:.3f}): {}'.format(epoch, best_score, ckpt_path))


# --------------------
# Logging utils
# --------------------

def make_writer(args):
    logdir = None
    if args.restore_ckpt:
        logdir = os.path.dirname(args.restore_ckpt)
    return SummaryWriter(log_dir=logdir)

def log_data_sample(batch_imgs, mode, writer, args, nrow=8):
    # load dataset mean
    mean, std = load_dataset_mean_and_std(args)
    mean, std = torch.from_numpy(mean), torch.from_numpy(std)

    # unnormalize and make grid for batch of shape (B,C,H,W) or sequence of shape (B,S,C,H,W)
    imgs = batch_imgs[:nrow,:args.k] if batch_imgs.ndim == 5 else batch_imgs[:nrow**2]
    imgs = imgs.flatten(0,1) if batch_imgs.ndim == 5 else imgs
    imgs = imgs.cpu() * std.view(1,-1,1,1) + mean.view(1,-1,1,1)
    imgs = make_grid(imgs, args.k if batch_imgs.ndim == 5 else nrow)

    # record
    writer.add_image(mode + '_data_sample', imgs, args.step)
    save_image(imgs, os.path.join(writer.log_dir, mode + '_data_sample.png'))

def log_confusion(preds, targets, writer, args):
    cm = confusion_matrix(targets, preds, labels=np.arange(args.n_classes))
    print(cm)
    print(np.histogram(preds, bins=args.n_classes)[0])
    fig = plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    writer.add_figure('confusion_matrix', fig, args.step)

def log_stats(preds, targets, train_loss, valid_loss, train_kappa, writer, args):
    log_confusion(preds, targets, writer, args)
    writer.add_histogram('preds_dist', preds, args.step)
    writer.add_scalar('loss_train', train_loss, args.step)
    writer.add_scalar('loss_valid', valid_loss, args.step)
    valid_kappa = cohen_kappa_score(targets, preds, weights='quadratic')
    print('validation kappa: {:.3f}'.format(valid_kappa))
    writer.add_scalar('kappa_train', train_kappa, args.step)
    writer.add_scalar('kappa_valid', valid_kappa, args.step)
    return valid_kappa


# --------------------
# Data
# --------------------

def load_df(args):
    df = pd.read_csv(args.df_path).set_index('image_id').sort_index()
    df.data_provider = df.data_provider.apply(lambda x: x.strip().lower())
    if 'debug' in args.model:
        img_idxs = sorted(list(set([os.path.basename(x).split('_')[0] for x in glob.glob(args.tiles_dir + '*.png')])))
        df = df.loc[img_idxs]
    return df

def remove_no_extractions_from_targets(df, args):
    with open(os.path.join(args.stats_dir, 'img_ids_with_no_tiles.txt'), 'r') as f:
        img_ids_with_no_tiles = f.read()
    img_ids_with_no_tiles = img_ids_with_no_tiles.split()
    return df.drop(index=img_ids_with_no_tiles, errors='ignore')

def split_targets(df, args):
    train_df, valid_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=args.seed)
    train_df = train_df.sort_index()
    valid_df = valid_df.sort_index()
    if 'debug' in args.model:
        valid_df = train_df.copy()
    return train_df, valid_df

def load_dataset_mean_and_std(args):
    if args.normalize_by_site:
        k_mean = np.loadtxt(os.path.join(args.stats_dir, 'karolinska_img_avg.txt'))
        k_std  = np.loadtxt(os.path.join(args.stats_dir, 'karolinska_img_std.txt'))

        r_mean = np.loadtxt(os.path.join(args.stats_dir, 'radboud_img_avg.txt'))
        r_std  = np.loadtxt(os.path.join(args.stats_dir, 'radboud_img_std.txt'))
        return (k_mean, r_mean), (k_std, r_std)
    else:
        mean = np.loadtxt(os.path.join(args.stats_dir, 'img_avg.txt'))
        std  = np.loadtxt(os.path.join(args.stats_dir, 'img_std.txt'))
        return mean, std

def blur(img, p=0.05):
    if np.random.rand() < p:
        img = img.filter(ImageFilter.BoxBlur(3))
    return img

def noise(img, p=0.1, var_limit=(0.2,0.45), mean=0):
    # N(0,sigma) where sigma in (10/255, 50/255) -- albumentations defaults
    if np.random.rand() < p:
        std = np.sqrt(np.random.uniform(*var_limit))
        img += np.random.normal(mean, std, size=(*img.size, 3))
        img = Image.fromarray(img.astype(np.uint8))
    return img

def make_ds_transform(mode, mean, std, args):
    if mode == 'train' and args.augment:
        transform = T.Compose([
            T.Lambda(blur), #lambda x: x.filter(ImageFilter.BoxBlur(3)) if np.random.rand() < 0.05 else x),
            T.Lambda(noise),#lambda x: x + (torch.rand() < 0.1) * torch.normal(torch.zeros_like(x), np.random.uniform(0.2, 0.45) * torch.ones_like(x))), # N(0,sigma) where sigma in (10/255, 50/255) -- albumentations defaults
            T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0.5, hue=0.1)], p=0.7),
            T.RandomApply([T.RandomAffine(degrees=(-45,45), shear=(-25,25))], p=0.2),
            T.RandomGrayscale(p=0.1),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
#            T.RandomErasing(p=0.2, scale=(0.02,0.3), ratio=(0.75, 1.75)),
            T.Normalize(mean, std),
            ])
    else:
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std),
            ])
    return transform

def make_dataset(df, args):
    # NOTE --  no transforms here
    if args.input == 'bags':
        return MILdatasetTopk(args.tiles_dir, df, args.k)
    else:
        return MILdataset(args.tiles_dir, df)

def make_dataloader(ds, mode, batch_size, shuffle, args, quiet=False):
    # load dataset mean
    mean, std = load_dataset_mean_and_std(args)

    # update dataset transform
    if args.normalize_by_site:
        ds.transform = {'karolinska': make_ds_transform(mode, mean[0], std[0], args),
                        'radboud': make_ds_transform(mode, mean[1], std[1], args)}
        ds.normalize_by_site = True
    else:
        ds.transform = make_ds_transform(mode, mean, std, args)
        ds.normalize_by_site = False

    # NOTE -- pin memory with bags training throws worker errors, so disable
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
            pin_memory=('cuda' in args.device and args.input != 'bags'), num_workers=args.n_workers)
    return dl

# --------------------
# Losses
# --------------------

def make_loss_fn(args):
    if args.loss_weights is not None:
        weight = torch.tensor(args.loss_weights)
        weight /= weight.sum()
        weight = weight.to(args.device)
        print('Using loss weights:', weight)

    if args.loss == 'ce':
        loss_fn = nn.CrossEntropyLoss(weight=weight if args.loss_weights is not None else None)
    elif args.loss == 'bce':
        loss_fn = partial(bce_loss, n_classes=args.n_classes)
        if args.loss_weights:
            loss_fn = partial(loss_fn, weight=weight)
    elif args.loss == 'masked_bce':
        loss_fn = partial(masked_bce_loss, n_classes=args.n_classes)
        if args.loss_weights:
            loss_fn = partial(loss_fn, weight=weight)
    elif args.loss == 'mse':
        loss_fn = partial(mse_loss, labels_mean=args.labels_mean, labels_std=args.labels_std)
    else:
        raise RuntimeError('Loss function not recognized.')

    return loss_fn

def mse_loss(logits, labels, labels_mean, labels_std):
    normalized_labels = (labels.float() - labels_mean) / labels_std
    return F.mse_loss(logits.squeeze(1), normalized_labels).mean(0)

def bce_loss(logits, labels, n_classes, weight=None):
    # e.g. for labels in {0,..,5}, label 4 -> [1,1,1,1,1,0] (active classes at and below label)
    labels = torch.arange(n_classes).repeat(len(labels), 1).to(labels.device)  <= labels[:,None]
    labels = labels.float()

    loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weight, reduction='none')
    return loss.sum(1).mean(0)

def masked_bce_loss(logits, labels, n_classes, weight=None):
    # e.g. for labels in {0,..,5}, label 4 -> [1,1,1,1,1,0] (active classes at and below label)
    labels = torch.arange(n_classes).repeat(len(labels), 1).to(labels.device)  <= labels[:,None]
    labels = labels.float()

    loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weight, reduction='none')

    # labels [1,1,0,0] -> mask [0,1,1,1]
    cumsum = labels.cumsum(1)                           # [1,1,0,0] -> [1,2,2,2]
    mask = cumsum == cumsum.max(1, keepdim=True).values # [1,2,2,2] -> [F,T,T,T]

#    assert torch.sum(mask * labels) == logits.shape[0] # should only overlap at the current label indicator, which is 1 in both label and mask

    loss *= mask
    return loss.sum(1).mean(0)

# --------------------
# MIL utils
# --------------------

def lexsort_and_index(probs, groups, k):
    order = np.lexsort((probs, groups))
    groups = groups[order]
    probs = probs[order]
    index = np.empty(len(groups), 'bool')
    index[-k:] = True
    index[:-k] = groups[k:] != groups[:-k]
    return groups, probs, order, index

def group_max(rankee, ranker, groups, nmax):
    # use ranker to rank rankee
    groups, probs, order, index = lexsort_and_index(ranker, groups, 1)
    out = np.full((nmax,rankee.shape[1]), np.nan)
    out[groups[index]] = rankee[order][index]  # reorder rankee based on order from lexsort (now order is given by ranker and matches order of groups ) then index
    return out

def group_argtopk(probs, groups, k=1):
    groups, probs, order, index = lexsort_and_index(probs, groups, k)
    return order[index]

def rank_tile_probs(probs, args):
    if args.loss == 'ce':
        # probs = CE output; shape (B, n_classes), each row is prob dist over the classes
        # rank based on highest class with prob > thresh (argmax) + prob under that class (when multiple tiles ranked in class 1, ranks tiles by prob under class 1)
        probs_argmax = np.argmax(probs * (probs >= args.clf_thresh), axis=1)
        max_probs = probs_argmax + probs[np.arange(len(probs)), probs_argmax]

    elif args.loss in ['bce', 'masked_bce']:
        # probs = masked BCE output; shape (B, n_classes), each row is prob dist for each class
        # NOTE -- if take np.argmax(probs >= thresh) np returns the first position, so eg:
        #           np.argmax([0.3, 0.6, 0.6] > 0.5) = 1; but want it to return 2 which is the highest class with prob > 0.5
        probs_argmax = np.max(np.tile(np.arange(args.n_classes), (probs.shape[0], 1)) * (probs >= args.clf_thresh), axis=1)
        max_probs = probs_argmax + probs[np.arange(len(probs)), probs_argmax]

    elif args.loss == 'mse':
        # probs = mse regression output; shape (B,1), each row is predicted target
        max_probs = probs.squeeze(1)

    else:
        raise RuntimeError('Cannot rank tile probabilities. Loss function not recognized.')

    return max_probs

def compute_slide_probs(tile_probs, slide_idxs, nmax, args):
    ranked_probs = rank_tile_probs(tile_probs, args)
    slide_probs = group_max(tile_probs, ranked_probs, slide_idxs, nmax)
    return slide_probs

# --------------------
# Inference
# --------------------

def process_model_output(logits, args):
    if args.loss == 'ce':
        return F.softmax(logits, dim=1)
    elif args.loss in ['bce', 'masked_bce']:
        return torch.sigmoid(logits)
    elif args.loss == 'mse':
        return logits * args.labels_std + args.labels_mean

@torch.no_grad()
def inference(encoder, model, dl, args, desc=''):
    model.eval()

    logits = []
    for x in tqdm(dl, desc=desc):
        x = x.to(args.device)
        if encoder is not None:
            x, _ = encoder(x)
        _, l = model(x)
        logits.append(l.cpu())
    logits = torch.cat(logits, dim=0)
    return logits

# --------------------
# Evaluation
# --------------------

def compute_preds(slide_probs, args):
    # compute preds from highest prob tile
    if args.loss == 'ce':
        # CE output is prob dist over the classes, so take argmax
        preds = np.argmax(slide_probs, axis=1)
    elif args.loss in ['bce', 'masked_bce']:
        # BCE output is prob dist under each class -> take the highest class with prob >= thresh
        preds = np.max(np.tile(np.arange(args.n_classes), (slide_probs.shape[0], 1)) * (slide_probs >= args.clf_thresh), axis=1)
    elif args.loss == 'mse':
        # regression output is the continuous target -> round to nearest int
        preds = np.round(slide_probs, 0).astype(np.int32)
        preds = np.clip(preds, 0, args.n_classes - 1)
        preds = preds.squeeze(1)
    else:
        raise RuntimeError('Cannot compute predications. Loss function not recognized.')

    return preds

def predict(model, dl, args, loss_fn=None, desc=''):
    dl.dataset.set_mode('eval')
    logits = inference(None, model, dl, args, desc)
    probs = process_model_output(logits, args).numpy()
    slide_probs = compute_slide_probs(probs, dl.dataset.slide_idxs, len(dl.dataset.df), args)
    preds = compute_preds(slide_probs, args)
    # if loss_fn provided (validation data), compute loss
    loss = None
    if loss_fn:
        # rank logits by probs depending on the loss function
        slide_logits = group_max(logits.numpy(), rank_tile_probs(probs, args), dl.dataset.slide_idxs, len(dl.dataset.df))
        loss = loss_fn(torch.from_numpy(slide_logits), torch.from_numpy(dl.dataset.df.isup_grade.values))
    return preds, slide_probs, loss  # slide level class preds and probs

def predict_bags(encoder, model, dl, args, loss_fn=None, desc=''):
    # rank all tiles in the dataset using encoder model and select topk
    dl.dataset.set_mode('eval')
    logits  = inference(None, encoder, dl, args, desc + ' - computing tile logits')
    probs = process_model_output(logits, args).numpy()
    ranked_probs = rank_tile_probs(probs, args)
    topk = group_argtopk(ranked_probs, dl.dataset.slide_idxs, args.k)
    dl.dataset.make_seq_data(topk, dl.dataset.slide_idxs)
    dl.dataset.set_mode('eval_seq')

    # run inference using bags model on the topk selected tiles for each slide
    slide_logits = inference(encoder, model, dl, args, desc + ' - computing slide logits')
    slide_probs = process_model_output(slide_logits, args).numpy()
    preds = compute_preds(slide_probs, args)

    # if loss_fn provided (validation data), compute loss
    loss = None
    if loss_fn:
        # rank logits by probs depending on the loss function
        loss = loss_fn(slide_logits, torch.from_numpy(dl.dataset.df.isup_grade.values))
    return preds, slide_probs, loss  # slide level class preds and probs

def evaluate(encoder, model, valid_dl, writer, args):
    predict_fn = partial(predict_bags, encoder) if encoder is not None else predict
    preds, probs, loss = predict_fn(model, valid_dl, args, desc='Evaluating')

    # compute metrics
    log_data_sample(next(iter(valid_dl)), 'valid', writer, args)
    targets = valid_dl.dataset.df.isup_grade
    cm = confusion_matrix(targets, preds)
    score = cohen_kappa_score(targets, preds, weights='quadratic')

    # save df with preds and probs
    valid_df = valid_dl.dataset.df
    valid_df['pred'] = preds
    # if classification save class probs
    probs_df = pd.DataFrame(probs.round(3), index=valid_df.index,
            columns=[f'prob_{i}' for i in range(probs.shape[1])] if probs.shape[1] > 1 else ['raw_pred'])
    valid_df = pd.concat([valid_df, probs_df], axis=1)
    valid_df.to_csv(os.path.join(writer.log_dir, 'valid_preds.csv'), index=True)

    # report
    print('Saved valid df with preds and probs.')
    print('Confusion matrix:\n', cm)
    print('Kappa = {:.4f}'.format(score))


# --------------------
# Training
# --------------------

def update_training_data(model, ds, args):
    ds.set_mode('eval')
    if args.resample_classes is not None:
        ds.resample(args.resample_classes)
    dl = make_dataloader(ds, 'train', args.valid_batch_size, shuffle=False, args=args, quiet=True) # no grads here so run on larger valid batch size
    logits = inference(None, model, dl, args, desc='Updating training data')
    probs = process_model_output(logits, args).numpy()
    ranked_probs = rank_tile_probs(probs, args)
    topk = group_argtopk(ranked_probs, dl.dataset.slide_idxs, args.k)
    ds.make_train_data(topk)
    ds.set_mode('train')

def train_epoch(encoder, model, dl, loss_fn, optimizer, args):
    model.train()

    cumloss = 0
    all_logits, all_targets = [], []
    with tqdm(total=len(dl), desc=f'Training epoch {args.epoch}') as pbar:
        for i, (x, y) in enumerate(dl):
            x, y = x.to(args.device), y.to(args.device)

            if encoder is not None:
                with torch.no_grad():
                    x, _ = encoder(x)

            _, logits = model(x)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cumloss += loss.item()
            all_logits.append(logits.detach().cpu())
            all_targets.append(y.cpu())

            args.step += 1
            pbar.set_postfix(loss = '{:.3f}'.format(cumloss / (i+1)))
            pbar.update()

    # output stats
    cumloss /= len(dl)
    probs = process_model_output(torch.cat(all_logits), args).numpy()
    slide_probs = compute_slide_probs(probs, np.arange(len(probs)), len(probs), args)
    preds = compute_preds(slide_probs, args)
    kappa = cohen_kappa_score(preds, torch.cat(all_targets).numpy(), weights='quadratic')

    return cumloss, kappa

def train(encoder, model, train_ds, valid_dl, loss_fn, optimizer, writer, args):
    for _ in range(args.epoch, args.epoch + args.n_epochs):
        args.epoch += 1

        update_training_data(encoder if encoder is not None else model, train_ds, args)
        train_dl = make_dataloader(train_ds, 'train', args.train_batch_size, shuffle=True, args=args, quiet=True)
        if args.step==0: log_data_sample(next(iter(train_dl))[0], 'train', writer, args)

        loss, kappa = train_epoch(encoder, model, train_dl, loss_fn, optimizer, args)

        if args.epoch % args.eval_interval == 0:
            predict_fn = partial(predict_bags, encoder) if encoder is not None else predict
            preds, _, valid_loss = predict_fn(model, valid_dl, args, loss_fn=loss_fn, desc='Evaluating')
            score = log_stats(preds, valid_dl.dataset.df.isup_grade.values, loss, valid_loss, kappa, writer, args)
            if score > args.best_score:
                args.best_score = score
                save_ckpt(model, optimizer, True, writer.log_dir, args)

        if args.epoch % args.save_interval == 0:
            save_ckpt(model, optimizer, False, writer.log_dir, args)


# --------------------
# Main
# --------------------

def main(args):
    # overwrite args if in debug mode
    if 'debug' in args.model:
        args.n_workers = 2
        args.train_batch_size = 32
        args.valid_batch_size = 32

    pprint.pprint(args.__dict__)

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
    if args.subset is not None: targets_df = targets_df.sample(args.subset)
    train_df, valid_df = split_targets(targets_df, args)
    args.labels_mean, args.labels_std = train_df.isup_grade.mean(), train_df.isup_grade.std()
    train_ds, valid_ds = make_dataset(train_df, args), make_dataset(valid_df, args)
    valid_dl = make_dataloader(valid_ds, 'valid', args.valid_batch_size, shuffle=False, args=args)

    # model
    #   classification vs regression output
    args.output_dim = args.n_classes if args.loss in ['ce', 'bce', 'masked_bce'] else 1
    if args.input == 'bags':
        encoder = Encoder(make_model(args.encoder, args), freeze=True) if 'debug' in args.encoder else \
                  Encoder.from_config(os.path.join(os.path.dirname(args.restore_encoder_ckpt), 'config.json'))
        encoder = encoder.to(args.device)
        args.embedding_dim = encoder.embedding_dim
        model = make_model(args.model, args)
    else:
        encoder = None
        model = make_model(args.model, args)
    model = model.to(args.device)

    # loss, optimizer
    loss_fn = make_loss_fn(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # restore
    if args.restore_encoder_ckpt: load_ckpt(args.restore_encoder_ckpt, encoder.model, None, args, update_args=False)
    if args.restore_ckpt:         load_ckpt(args.restore_ckpt, model, optimizer, args, update_args=True)

    # writer
    writer = make_writer(args)
    save_json(args.__dict__, os.path.join(writer.log_dir, 'config.json'))

    # train
    if args.train:
        train(encoder, model, train_ds, valid_dl, loss_fn, optimizer, writer, args)

    # evaluate
    if args.evaluate:
        evaluate(encoder, model, valid_dl, writer, args)

if __name__ == '__main__':
    import signal, sys
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    args = parser.parse_args()
    main(args)
