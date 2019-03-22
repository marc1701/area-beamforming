import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def binarise(doa_data, step=0.02):

    # set up output
    frames = np.arange(0, doa_data[:,1].max(), step)
    angles = np.unique(doa_data[:,2:], axis=0)
    array = np.zeros((len(frames), len(angles)), dtype=bool)

    # find indices where source number changes
    indices = np.where(np.diff(doa_data[:,0]))[0]

    # add NaN values in to array at these indices
    # *** will this work with simultaneous sources? ***
    for n, idx in enumerate(indices):
        idx += 2*n
        nan_array = np.array([[np.nan, doa_data[idx,1], np.nan, np.nan],
                              [np.nan, doa_data[idx+1,1], np.nan, np.nan]])
        doa_data = np.insert(doa_data, idx+1, nan_array, axis=0)
    zero_nan_array = np.array([[np.nan, 0, np.nan, np.nan],
                               [np.nan, doa_data[0,1], np.nan, np.nan]])
    doa_data = np.insert(doa_data, 0, zero_nan_array, axis=0)

    # binarise angle activation data
    for t in frames:
        angle = doa_data[np.where(t<doa_data[:,1])[0].min(),2:]
        angle_index = (angles == angle).all(axis=1).nonzero()
        time_index = np.where(frames == t)[0][0]

        if np.size(angle_index) > 0:
            array[time_index, angle_index] = True

    return array, angles


def frame_recall(ref_array, pred_array):

    D_R = ref_array.sum(axis=1)
    D_P = pred_array.sum(axis=1)

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


def doa_error(ref_array, ref_angles, pred_array, pred_angles):

    total_error = 0.0

    for ref, pred in zip(ref_array, pred_array):

        doa_r = ref_angles[np.where(ref)[0]]
        doa_p = pred_angles[np.where(pred)[0]]

        frame_error = angle_error(doa_r, doa_p)

        total_error += frame_error

    return total_error / np.sum(pred_array)
