# Behavior-aware Variational Autoencoder (BVAE)

## Contents
- Dataset
- Codes
- Log 

## Environment

- python = 3.6.13
- tensorflow == 1.15.0
- scipy = 1.3.1
- numpy = 1.19.5

## How to Run the Codes

1. python run.py

2.Or:
#### On the JD dataset

- python train_2.py --dataset JD --batch_size 500 --user_num 10690 --item_num 13465 --behavior_num 2 --task_num 2 --hiddenDim 100 --lr_rate 0.001
- python train_3.py --dataset JD --batch_size 500 --user_num 10690 --item_num 13465 --behavior_num 3 --task_num 3 --hiddenDim 100 --lr_rate 0.001
- python train_4.py --dataset JD --batch_size 500 --user_num 10690 --item_num 13465 --behavior_num 4 --task_num 4 --hiddenDim 100 --lr_rate 0.001

#### On the UB dataset
- python train_2.py --dataset UB --batch_size 500 --user_num 20443 --item_num 30947 --behavior_num 2 --task_num 2 --hiddenDim 100 --lr_rate 0.001
- python train_3.py --dataset UB --batch_size 500 --user_num 20443 --item_num 30947 --behavior_num 3 --task_num 3 --hiddenDim 100 --lr_rate 0.001
- python train_4.py --dataset UB --batch_size 500 --user_num 20443 --item_num 30947 --behavior_num 4 --task_num 4 --hiddenDim 100 --lr_rate 0.001
