import torch
from numba import jit

def f_weighted_bce_loss(hat_y, y):
    """
    computes a weighted BCE 

    Args:
        hat_y
        y
    Returns:
        loss
    """

    N = y.shape[-1]
    nb_elem = N**2
    nb_one = y.sum()
    nb_zero = nb_elem - nb_one

    hat_y = torch.clamp(hat_y, 1e-7, 1 - 1e-7)
    one_contrib = (nb_zero/nb_elem) * y * hat_y.log() # --- weight_one: if they are more zero than one -> over-weight the one loss
    zero_contrib = (nb_one/nb_elem) * (1 - y) * (1 - hat_y).log() # --- if they are more one than zero -> over-weight the zero loss
    loss = - torch.mean(one_contrib + zero_contrib)
    return loss


from scipy.signal import find_peaks
import numpy as np

def f_get_peaks(data_v, config_d, step_sec):
    """
    detect peaks of a function, peaks are defined as local maxima above a threshold

    Args:
        data_v
        config_d
        step_sec

    Returns:
        est_boundary_frame_v
    """

    #param_Thalf = 10
    #param_tau = 1.35
    #param_distance = 7
    param_Thalf = int(np.round(config_d['peak_mean_Ldemi_sec']/step_sec))
    param_distance = int(np.round(config_d['peak_distance_sec']/step_sec))
    param_tau = config_d['peak_threshold']
    
    nb_frame = len(data_v)
    
    # --- Compute peak_to_mean ratio
    peak_to_mean_v = np.zeros((nb_frame))
    for nu in range(0, nb_frame):
        sss = max(0, nu - param_Thalf)
        eee = min(nu + param_Thalf+1, nb_frame)
        local_mean = np.sum(data_v[sss:eee]) / (eee-sss)
        peak_to_mean_v[nu] = data_v[nu] / local_mean
    
    # --- Find peaks 
    peaks, _ = find_peaks(peak_to_mean_v, distance = param_distance)
    
    # --- Above threshold tau
    above_treshold = np.where(peak_to_mean_v[peaks] >= param_tau)[0]
    est_boundary_frame_v = peaks[above_treshold]
    
    return est_boundary_frame_v