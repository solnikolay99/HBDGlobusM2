# ******************************************************************************
# @author: L. I. Nurtdinova
# ******************************************************************************

import glob
import os

import numpy as np


def name_reader(directory_path, pattern):
    return sorted(glob.glob(directory_path + '/' + pattern))[:]


def save(folder, name):
    directory_path = os.getcwd() + folder
    limit = 500000
    file_paths = name_reader(directory_path, 'coorx*')
    coorx = np.concatenate([np.load(file_path)[:limit, :] for file_path in file_paths], axis=0)
    file_paths = name_reader(directory_path, 'coory*')
    coory = np.concatenate([np.load(file_path)[:limit, :] for file_path in file_paths], axis=0)
    file_paths = name_reader(directory_path, 'coorz*')
    coorz = np.concatenate([np.load(file_path)[:limit, :] for file_path in file_paths], axis=0)
    ln = coory.shape[0]
    print(ln)
    timestart, timelimit = 75, 100
    timestart, timelimit = 100, 125
    coorx = coorx[:, timestart:timelimit]
    coory = coory[:, timestart:timelimit]
    coorz = coorz[:, timestart:timelimit]

    m = 1
    x, y = 150, 50
    grd = np.zeros((x // m, y // m, y // m))
    combined = np.column_stack(
        (np.floor(coorx).astype(int).ravel(), np.floor(coory).astype(int).ravel(), np.floor(coorz).astype(int).ravel()))
    coordinates, values = np.unique(combined, axis=0, return_counts=True)
    grd[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] = values
    grd = np.rot90(grd, k=1, axes=(0, 1))
    np.save(name + '.npy', grd)


save('/001', 'data/001')
save('/002', 'data/002')
save('/003', 'data/003')
save('/004', 'data/004')
save('/005', 'data/005')
