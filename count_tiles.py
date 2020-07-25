import os
import glob
import json
import pprint
import argparse
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('--tiles_dir', type=str, default='./data/train_images/tiles/')

def count(args):
    tiles = glob.glob(args.tiles_dir + '*.png')
    counter = Counter([os.path.basename(t).split('_')[0] for t in tiles])

    with open('tile_counts.json', 'w') as f:
        json.dump(counter, f, sort_keys=True)

    pprint.pprint(counter)


if __name__ == '__main__':
    args = parser.parse_args()
    count(args)
