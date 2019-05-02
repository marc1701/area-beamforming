import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def binarise(doa_data, step=0.02):

    reformat_doa_data = np.zeros((len(np.unique(doa_data[:,0])), 4))

    for i, obj in enumerate(np.unique(doa_data[:,0])):

        obj_start = min(np.where(doa_data[:,0] == obj)).min()
        obj_stop = min(np.where(doa_data[:,0] == obj)).max()

        reformat_doa_data[i] = np.array([doa_data[obj_start,1],
                                         doa_data[obj_stop,1],
                                         doa_data[obj_start,3],
                                         doa_data[obj_stop,2]])

    doa_data = reformat_doa_data

    # set up output
    frames = np.arange(0, doa_data[:,1].max(), step)
    angles = np.unique(doa_data[:,2:], axis=0)
    array = np.zeros((len(frames), len(angles)), dtype=bool)

    for t in frames:
        a = t > doa_data[:,0] # t > onset
        b = t < doa_data[:,1] # t < offset
        c = a == b

        f_angles = doa_data[c,2:]

        angle_indices = [(angles == angle).all(axis=1).nonzero()[0][0]
            for angle in f_angles]
        time_index = np.where(frames == t)[0][0]

        for angle_index in angle_indices:
            if np.size(angle_index) > 0:
                array[time_index, angle_index] = True

    return array, angles


def frame_recall(ref_doa_data, pred_doa_data):
    ref_array, _ = binarise(ref_doa_data)
    pred_array, _ = binarise(pred_doa_data)

    D_R = ref_array.sum(axis=1)
    D_P = pred_array.sum(axis=1)

    # match lengths of reference and prediction vectors ...
    if len(D_R) > len(D_P):
        pad_len = len(D_R) - len(D_P)
        pad = np.zeros((pad_len))
        D_P = np.concatenate((D_P, pad))
    elif len(D_R) < len(D_P):
        pad_len = len(D_P) - len(D_R)
        pad = np.zeros((pad_len))
        D_R = np.concatenate((D_R, pad))

    # ... or this comparison doesn't work properly
    return (D_R == D_P).sum() / len(ref_array)


def angular_dist(sph_angle1, sph_angle2):

    delta1 = np.deg2rad(sph_angle1[1])
    delta2 = np.deg2rad(sph_angle2[1])
    alpha1 = np.deg2rad(sph_angle1[0])
    alpha2 = np.deg2rad(sph_angle2[0])

    return np.rad2deg(np.arccos(
            np.sin(delta1)*np.sin(delta2) +
            np.cos(delta1)*np.cos(delta2)*np.cos(alpha1-alpha2)))


def angle_error(ref_angles, pred_angles):

    # make matrix of angular distances between all pairwise angle combinations
    distance_matrix = cdist(ref_angles, pred_angles, angular_dist)

    # pair up each predicted angle to nearest reference angle
    a, b = linear_sum_assignment(distance_matrix)

    return distance_matrix[a,b].sum()


def doa_error(ref_doa_data, pred_doa_data):

    ref_array, ref_angles = binarise(ref_doa_data)
    pred_array, pred_angles = binarise(pred_doa_data)

    total_error = 0.0

    for ref, pred in zip(ref_array, pred_array):

        doa_r = ref_angles[np.where(ref)[0]]
        doa_p = pred_angles[np.where(pred)[0]]

        frame_error = angle_error(doa_r, doa_p)

        total_error += frame_error

    return total_error / np.sum(pred_array)
