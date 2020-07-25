# Prostate Cancer Grade Assessment

This repo summarizes my work on the Kaggle challenge on [Prostate Cancer Grade Assessment](https://www.kaggle.com/c/prostate-cancer-grade-assessment). My goal was to reimplement two papers and apply the methods described on a multi-class classification problem:
* Multiple Instance Learning (MIL) approach from Campenella et al. 2019. [Clinical-grade computational pathology using weakly supervised deep learning on whole slide images](https://www.nature.com/articles/s41591-019-0508-1)
* Attention-based  MIL approach of Ilse et al. 2019. [Attention-based Deep Multiple Instance Learning](https://arxiv.org/abs/1802.04712)

The approach here is slightly different, as both papers address a binary classification problem and my implementation here is for a multi-class problem, which I attempt under both a classification and regression definition. Unfortunately I joined the competition late and had limited access to a GPU, so no results are included.

## Data

Dataset can be downloaded from Kaggle here: https://www.kaggle.com/c/prostate-cancer-grade-assessment/data

## Usage

##### To extract tiles:
* default extracts 224x224 from the level 1 tiff images
* tile overlap controls % overlap of the tiles
* function also computes and stores image mean and std

```
python tile.py
    --tiff_dir  ./data/train_images/
    --tiles_dir ./data/train_images/tiles_level_1_overlap_0/
    --df_path   ./data/train.csv
    --overlap   0
```


##### To train and evaluate model on tiles:
```
python train.py
    --train           # perform training
    --evaluate        # store predictions data frame when training is complete
    --cuda 0          # which GPU to train/evaluate on
    --model resnet34  # choice from resnet34 and efficientnet
    --pretrained      # whether to use ImageNet pretrained weights
    --loss ce         # choice from mse, ce, bce, masked_bce
    --input tiles     # run model on tiles
```
Additional flags for number of epochs, learning rate, train and valid batch size, under- or over-sampling of classes, site-specific image normalization, class weights, data augmentation, etc.

##### To train and evaluate model on bags:
```
python train.py
    --train                  # perform training
    --evaluate               # store predictions data frame when training is complete
    --cuda 0                 # which GPU to train/evaluate on
    --encoder resnet34       # choice from resnet34 and efficientnet
    --restore_encoder_ckpt [path to saved checkpoint and config of the trained encoder model]
    --model tanh-attn-small  # bags model to use; choice from rnn, tanh-attn-small/big, mha
    --loss ce                # choice from mse, ce, bce, masked_bce
    --input bags             # run model on bags of tiles where topk
    --k                      # number of tiles with highest probability to include in each slide bag
```

##### Evaluation output:
* `--evaluate` outputs a confusion matrix and saves a `predictions.csv` with slide name, slide target, model prediction and class probabilities (depending on loss function) for each slide in the test data.
* if `--valid_df_path` is not specified, training data frame is randomly split into training and validation set.

## Requirements
* python 3.8
* pytorch 1.5.1
* scikit-image (used for loading tiff images and extracting tiles)
* scikit-learn
* numpy
* pandas
* matplotlib
* seaborn
 
## References
* https://github.com/AMLab-Amsterdam/AttentionDeepMIL/
* https://github.com/MSKCC-Computational-Pathology/MIL-nature-medicine-2019