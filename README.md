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

2. Or:
#### On the JD dataset

- python train_2.py --dataset JD --batch_size 500 --user_num 10690 --item_num 13465 --behavior_num 2 --task_num 2 --hiddenDim 100 --lr_rate 0.001
- python train_3.py --dataset JD --batch_size 500 --user_num 10690 --item_num 13465 --behavior_num 3 --task_num 3 --hiddenDim 100 --lr_rate 0.001
- python train_4.py --dataset JD --batch_size 500 --user_num 10690 --item_num 13465 --behavior_num 4 --task_num 4 --hiddenDim 100 --lr_rate 0.001

#### On the UB dataset
- python train_2.py --dataset UB --batch_size 500 --user_num 20443 --item_num 30947 --behavior_num 2 --task_num 2 --hiddenDim 100 --lr_rate 0.001
- python train_3.py --dataset UB --batch_size 500 --user_num 20443 --item_num 30947 --behavior_num 3 --task_num 3 --hiddenDim 100 --lr_rate 0.001
- python train_4.py --dataset UB --batch_size 500 --user_num 20443 --item_num 30947 --behavior_num 4 --task_num 4 --hiddenDim 100 --lr_rate 0.001

## ACM Reference Format:
Qianzhen Rao, Yang Liu, Weike Pan, and Ming Zhong. 2023. BVAE: Behavior-aware Variational Autoencoder for Multi-Behavior
Multi-Task Recommendation. In Seventeenth ACM Conference on Recommender Systems (RecSys ’23)

## Note
We made an error in the data statistics table in our paper: the total number of items in the UB dataset is 30,947, and the 30,743 written in the paper is the number of items in the training set. Similarly, the total number of purchase interactions in the UB dataset should be 133,708. Sincerely apologize.
