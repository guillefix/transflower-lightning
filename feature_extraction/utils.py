from math import floor
from scipy.signal import resample
from scipy.interpolate import interp1d
import numpy as np

def distribute_tasks(tasks, rank, size):
    num_tasks = len(tasks)
    num_tasks_per_job = num_tasks//size
    tasks = list(range(rank*num_tasks_per_job,(rank+1)*num_tasks_per_job))
    if rank < num_tasks%size:
        tasks.append(size*num_tasks_per_job+rank)
    return tasks

def ResampleLinear1D(original, targetLen):
    index_arr = np.linspace(0, len(original)-1, num=targetLen, dtype=np.float)
    index_floor = np.array(index_arr, dtype=np.int) #Round down
    index_ceil = index_floor + 1
    index_rem = index_arr - index_floor #Remain

    val1 = original[index_floor]
    val2 = original[index_ceil % len(original)]
    interp = val1 * np.expand_dims(1.0-index_rem,1) + val2 * np.expand_dims(index_rem,1)
    assert(len(interp) == targetLen)
    return interp


def downsample_signal(original,ratio):
    # ratio is the ratio between the time step in the original and in the new one (so it should be > 1 for downsampling)
    old_indices = np.arange(len(original), dtype=np.int32)
    new_indices = (old_indices // ratio).astype(np.int32)
    M = np.zeros((int(len(original)//ratio)+1,len(original)), dtype=np.float32)
    M[new_indices,old_indices] = 1
    w = np.sum(M,1)
    w = np.expand_dims(w,1)
    # print(M.shape)
    return np.matmul(M,original)/w
