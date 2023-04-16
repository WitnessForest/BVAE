import random
import logging
import os
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
from dataset import loadTrainData, loadTestData
import evaluation
from model import VAE
from scipy import sparse
from Params import args

# 不打印低于error级别的错误
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train(args, model, sess):
    stop_count, anneal_cap, update_count = 0, 0.2, 0.0
    topN = [5, 10, 15, 20]
    best_iter, best_ndcg5 = 0, 0.
    best_ndcg5_p, best_ndcg5_cl = 0., 0.
    best_iter_p, best_iter_cl = 0, 0

    # read train data
    targetData_matrix, targetDict = loadTrainData(args=args, i=0)
    users_set = set(targetDict.keys())
    clickData_matrix, clickDict = loadTrainData(args=args, i=1)
    users_set = users_set.union(set(clickDict.keys()))
    userList_train = sorted(list(users_set))

    # read test data
    userList = []
    targetTestDict = loadTestData(args=args, i=0)
    userList.append(sorted(set(targetTestDict.keys())))
    clickTestDict = loadTestData(args=args, i=1)
    userList.append(sorted(set(clickTestDict.keys())))

    for epoch in range(args.epoch):
        # train
        random.shuffle(userList_train)
        loss = 0.

        for bnum, st_idx in enumerate(range(0, len(userList_train), args.batch_size)):
            end_idx = min(st_idx + args.batch_size, len(userList_train))
            batchIndex = userList_train[st_idx: end_idx]

            if args.total_anneal_steps > 0:
                anneal = min(anneal_cap, 1. * update_count / args.total_anneal_steps)
            else:
                anneal = anneal_cap

            B1 = targetData_matrix[batchIndex]
            if sparse.isspmatrix(B1):
                B1 = B1.toarray()
            B2 = clickData_matrix[batchIndex]
            if sparse.isspmatrix(B2):
                B2 = B2.toarray()
            D = B1 + B2
            D = (D != 0).astype(np.float64)

            # target loss
            feed_dict = {model.input_ph_union: D, model.input_ph_behavior1: B1, model.input_ph_behavior2: B2,
                         model.keep_prob_ph: 0.5, model.anneal_ph: anneal, model.is_training_ph: 1}
            _, e_loss = sess.run([model.train_op, model.loss], feed_dict=feed_dict)

            update_count += 1
            loss += e_loss

        # test
        taskName = ['Purchase:  ', 'Click:     ', 'Favourite: ', 'Cart:      ']
        precision, recall, f1, ndcg, one_call, mrr = [], [], [], [], [], []
        for task_i in range(args.task_num):
            precision_t, recall_t, f1_t, ndcg_t, one_call_t, mrr_t = [0., 0., 0., 0.], [0., 0., 0., 0.], \
                                                                     [0., 0., 0., 0.], [0., 0., 0., 0.], \
                                                                     [0., 0., 0., 0.], [0., 0., 0., 0.]
            userList_test = userList[task_i]
            for bnum, st_idx in enumerate(range(0, len(userList_test), args.batch_size)):
                end_idx = min(st_idx + args.batch_size, len(userList_test))
                batchIndex = userList_test[st_idx: end_idx]
                B1 = targetData_matrix[batchIndex]
                if sparse.isspmatrix(B1):
                    B1 = B1.toarray()
                B2 = clickData_matrix[batchIndex]
                if sparse.isspmatrix(B2):
                    B2 = B2.toarray()
                D = B1 + B2
                D = (D != 0).astype(np.float64)

                testDict_batch = []
                feed_dict = {model.input_ph_union: D, model.input_ph_behavior1: B1, model.input_ph_behavior2: B2}
                allRatings_batch = sess.run(model.logits_target[task_i], feed_dict=feed_dict)

                if task_i == 0:
                    allRatings_batch[B1.nonzero()] = -np.inf
                    for i in batchIndex:
                        testDict_batch.append(targetTestDict[i])
                if task_i == 1:
                    allRatings_batch[B2.nonzero()] = -np.inf
                    for i in batchIndex:
                        testDict_batch.append(clickTestDict[i])

                _, predictedIndices = sess.run(model.top_k, feed_dict={model.prediction_top_k: allRatings_batch,
                                                                       model.scale_top_k: topN[-1]})

                precision_batch, recall_batch, f1_batch, ndcg_batch, one_call_batch, mrr_batch = evaluation.computeTopNAccuracy(
                    testDict_batch, predictedIndices, topN)

                for index in range(len(topN)):
                    precision_t[index] += precision_batch[index] / len(userList_test)
                    recall_t[index] += recall_batch[index] / len(userList_test)
                    f1_t[index] += f1_batch[index] / len(userList_test)
                    ndcg_t[index] += ndcg_batch[index] / len(userList_test)
                    one_call_t[index] += one_call_batch[index] / len(userList_test)
                    mrr_t[index] += mrr_batch[index] / len(userList_test)

            precision.append(precision_t)
            recall.append(recall_t)
            f1.append(f1_t)
            ndcg.append(ndcg_t)
            one_call.append(one_call_t)
            mrr.append(mrr_t)

        print('Epoch: %d Loss: %.4f' % (epoch, loss))
        logger.info('Epoch: %d Loss: %.4f' % (epoch, loss))
        for i in range(args.task_num):
            print(taskName[i] + 'NDCG@5: %.4f\tNDCG@10: %.4f\tNDCG@15: %.4f\tNDCG@20: %.4f' % (
                ndcg[i][0], ndcg[i][1], ndcg[i][2], ndcg[i][3]))
            print(taskName[i] + 'HR@5:   %.4f\tHR@10:   %.4f\tHR@15:   %.4f\tHR@20:   %.4f' % (
                one_call[i][0], one_call[i][1], one_call[i][2], one_call[i][3]))
            print(taskName[i] + 'PRE@5:  %.4f\tPRE@10:  %.4f\tPRE@15:  %.4f\tPRE@20:  %.4f' % (
                precision[i][0], precision[i][1], precision[i][2], precision[i][3]))
            print(taskName[i] + 'REC@5:  %.4f\tREC@10:  %.4f\tREC@15:  %.4f\tREC@20:  %.4f' % (
                recall[i][0], recall[i][1], recall[i][2], recall[i][3]))
            print(taskName[i] + 'F1@5:   %.4f\tF1@10:   %.4f\tF1@15:   %.4f\tF1@20:   %.4f' % (
                f1[i][0], f1[i][1], f1[i][2], f1[i][3]))
            logger.info(taskName[i] + 'NDCG@5: %.4f\tNDCG@10: %.4f\tNDCG@15: %.4f\tNDCG@20: %.4f' % (
                ndcg[i][0], ndcg[i][1], ndcg[i][2], ndcg[i][3]))
            logger.info(taskName[i] + 'HR@5:   %.4f\tHR@10:   %.4f\tHR@15:   %.4f\tHR@20:   %.4f' % (
                one_call[i][0], one_call[i][1], one_call[i][2], one_call[i][3]))
            logger.info(taskName[i] + 'PRE@5:  %.4f\tPRE@10:  %.4f\tPRE@15:  %.4f\tPRE@20:  %.4f' % (
                precision[i][0], precision[i][1], precision[i][2], precision[i][3]))
            logger.info(taskName[i] + 'REC@5:  %.4f\tREC@10:  %.4f\tREC@15:  %.4f\tREC@20:  %.4f' % (
                recall[i][0], recall[i][1], recall[i][2], recall[i][3]))
            logger.info(taskName[i] + 'F1@5:   %.4f\tF1@10:   %.4f\tF1@15:   %.4f\tF1@20:   %.4f' % (
                f1[i][0], f1[i][1], f1[i][2], f1[i][3]))

        # early stop
        ndcg5 = np.array(ndcg)[:, 0]
        if best_ndcg5 < sum(ndcg5):
            best_iter = epoch
            stop_count = 0
            best_ndcg5 = sum(ndcg5)
        else:
            stop_count += 1
        print("BestEpoch: %d BestNDCG@5: %.4f  StopCount: %d" % (best_iter, best_ndcg5, stop_count))
        logger.info("BestEpoch: %d BestNDCG@5: %.4f  StopCount: %d" % (best_iter, best_ndcg5, stop_count))
        if stop_count >= args.early_stop:
            break

        if best_ndcg5_p < ndcg5[0]:
            best_iter_p = epoch
            best_ndcg5_p = ndcg5[0]
        print("Best NDCG@5_purchase = %.4f when %d-th epoch" % (best_ndcg5_p, best_iter_p))
        logger.info("Best NDCG@5_purchase = %.4f when %d-th epoch" % (best_ndcg5_p, best_iter_p))
        if best_ndcg5_cl < ndcg5[1]:
            best_iter_cl = epoch
            best_ndcg5_cl = ndcg5[1]
        print("Best NDCG@5_click = %.4f when %d-th epoch" % (best_ndcg5_cl, best_iter_cl))
        logger.info("Best NDCG@5_click = %.4f when %d-th epoch" % (best_ndcg5_cl, best_iter_cl))

    # 存最优
    print("End. Best Iteration %d: NDCG@5 = %.4f " % (best_iter, best_ndcg5))
    logger.info("End. Best Iteration %d: NDCG@5 = %.4f " % (best_iter, best_ndcg5))

    print("Best NDCG@5_purchase = %.4f when %d-th epoch" % (best_ndcg5_p, best_iter_p))
    logger.info("Best NDCG@5_purchase = %.4f when %d-th epoch" % (best_ndcg5_p, best_iter_p))
    print("Best NDCG@5_click = %.4f when %d-th epoch" % (best_ndcg5_cl, best_iter_cl))
    logger.info("Best NDCG@5_click = %.4f when %d-th epoch" % (best_ndcg5_cl, best_iter_cl))


if __name__ == '__main__':

    model = VAE(args=args)
    model.build_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S.%f')
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    # 记录器Logger：负责日志的记录
    logger = logging.getLogger('Log')
    # DEBUG < INFO < WARNING < ERROR < CRITICAL, 低于该级别的不会被输出
    logger.setLevel(logging.INFO)

    log_dir = './Log/' + args.dataset + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # FileHandler 文件处理器，将消息发送到文件
    fh = logging.FileHandler(filename=os.path.join(log_dir, "BVAE_%s_behavior%d_batch%d_hidden%s_lr%.4f_%s.res" % (
        args.dataset, args.behavior_num, args.batch_size, args.hiddenDim, args.lr_rate, timestamp)), mode='w')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    logger.info(args)

    start_time = time.time()
    timestamp = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
    logger.info('Start time: %s' % timestamp)

    train(args, model, sess)

    end_time = time.time()
    timestamp = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
    logger.info('End time: %s' % timestamp)

    use_time = time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))
    print('Use time: %s' % use_time)
    logger.info('Use time: %s' % use_time)
