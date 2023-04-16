import argparse
import ast


def parse_args():
    parser = argparse.ArgumentParser()
    # data parameters
    parser.add_argument('--path', type=str, default='Dataset')
    parser.add_argument('--dataset', type=str, default='JD')
    parser.add_argument('--user_num', type=int, default=10690)
    parser.add_argument('--item_num', type=int, default=13465)
    # JD
    # args.user = 10690
    # args.item = 13465
    # UB
    # args.user = 20443
    # args.item = 30947
    parser.add_argument('--behavior_num', type=int, default=2)
    parser.add_argument('--task_num', type=int, default=2)
    # args.behavior <= 4
    parser.add_argument('--train', type=str, default=['target_train', 'click_train', 'favourite_train', 'cart_train'])
    parser.add_argument('--test', type=str, default=['target_test', 'click_test', 'favourite_test', 'cart_test'])
    parser.add_argument('--valid', type=str, default=['target_valid', 'click_valid', 'favourite_valid', 'cart_valid'])
    # model parameters
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--lr_rate', type=float, default=1e-3)
    parser.add_argument('--reg_scale', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--hiddenDim', type=int, default=100)
    # mult-task parameters
    parser.add_argument('--share_expert_num', type=int, default=1)
    parser.add_argument('--specific_expert_num', type=int, default=1)
    parser.add_argument('--num_levels', type=int, default=1)
    # training parameters
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--early_stop', type=int, default=500)
    parser.add_argument('--topk', type=int, default=20)
    # other parameters
    parser.add_argument('--total_anneal_steps', type=int, default=200000)
    parser.add_argument('--is_train', type=ast.literal_eval, default=True)
    parser.add_argument('--random_seed', type=int, default=724)
    return parser.parse_args()


args = parse_args()
