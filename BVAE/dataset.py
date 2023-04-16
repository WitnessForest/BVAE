import numpy as np
import pandas as pd
from scipy import sparse


def loadTrainData(args, i):
    file = args.path + '/' + args.dataset + '/' + args.train[i]
    df = pd.read_csv(file, sep=' ', names=['uid', 'iid'])
    df = df.sort_values("uid")

    rows, cols = df['uid'], df['iid']
    usersNum, itemsNum = args.user_num + 1, args.item_num + 1
    trainData = sparse.csr_matrix(
        (np.ones_like(rows), (rows, cols)), dtype='float64', shape=(usersNum, itemsNum))

    trainDict = df.groupby('uid')['iid'].apply(list).to_dict()
    return trainData, trainDict


def loadTestData(args, i):
    file = args.path + '/' + args.dataset + '/' + args.test[i]
    df = pd.read_csv(file, sep=' ', names=['uid', 'iid'])
    df = df.sort_values("uid")

    testDict = df.groupby('uid')['iid'].apply(list).to_dict()
    return testDict
