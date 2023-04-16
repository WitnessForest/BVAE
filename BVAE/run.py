import multiprocessing
import os

# cores = int(multiprocessing.cpu_count() / 2)

dataset_set = [
    ['JD', 10690, 13465, 2],
    ['JD', 10690, 13465, 3],
    ['JD', 10690, 13465, 4],
    ['UB', 20443, 30947, 2],
    ['UB', 20443, 30947, 3],
    ['UB', 20443, 30947, 4],
]

cuda_id = 0
# run_num = 3
run_num = 1

batch_size_set = [500]
hiddenDim_set = [100]  # [50, 100, 150, 200]
lr_rate_set = [1e-3]  # [1e-2, 1e-3, 1e-4]


def exec_command(arg):
    os.system(arg)


# params coarse tuning function
def coarse_tune():
    command = []
    for index in range(run_num):
        print(index + 1)
        for dataset in dataset_set:
            for batch_size in batch_size_set:
                for hiddenDim in hiddenDim_set:
                    for lr_rate in lr_rate_set:
                        cmd = '  CUDA_VISIBLE_DEVICES=' + str(cuda_id) + ' python train_' + str(dataset[3]) + '.py ' \
                              + '--dataset ' + dataset[0] + ' --batch_size ' + str(batch_size) \
                              + ' --user_num ' + str(dataset[1]) + ' --item_num ' + str(dataset[2]) \
                              + ' --behavior_num ' + str(dataset[3]) + ' --task_num ' + str(dataset[3]) \
                              + ' --hiddenDim ' + str(hiddenDim) + ' --lr_rate ' + str(lr_rate)
                        print(cmd)
                        command.append(cmd)
    print('\n')

    pool = multiprocessing.Pool(processes=2)
    for cmd in command:
        pool.apply_async(exec_command, (cmd,))
    pool.close()
    pool.join()


if __name__ == '__main__':
    coarse_tune()
