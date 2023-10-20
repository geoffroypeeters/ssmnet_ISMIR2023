from __future__ import annotations

import numpy as np
from scipy.signal import convolve
from scipy.signal import find_peaks

from typing import Tuple

import torch
import librosa


def f_weighted_bce_loss(hat_y, y):
    """
    Computes a weighted BCE 

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


def f_get_peaks(data_v, config_d, step_sec):
    """
    Detect peaks of a function, peaks are defined as local maxima above a threshold

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


def f_extract_feature(audio_v: np.ndarray,
                    sr_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute audio features

    Args:
        audio_v (np.ndarray)
        sr_hz (float)

    Returns:
        logmel_m    (np.ndarray)
        time_sec_v  (np.ndarray)
    """

    # --- 1) compute mel-spectrogram
    mel_m = librosa.feature.melspectrogram(y=audio_v, sr=sr_hz, n_mels=80, fmax=8000)
    # --- 2) convert to log
    gamma = 100
    logmel_m = np.log(1 + gamma * mel_m)
    time_sec_v = librosa.frames_to_time(frames=np.arange(0, logmel_m.shape[1]))

    logmel_m -= logmel_m.mean(axis=1, keepdims=True)
    logmel_m /= (logmel_m.std(axis=1, keepdims=True) + np.finfo(float).eps)

    return logmel_m, time_sec_v


def f_reduce_time(data_m: np.ndarray,
                time_sec_v: np.ndarray,
                step_target_sec: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce the time axis of data_m using an averaging method

    Args:
        data_m
        time_sec_v
        step_target_sec

    Returns:
        data_sync_m
        time_sync_sec_v
    """

    # --- 3) reduce time-step
    # --- https://librosa.org/doc/main/generated/librosa.stft.html#librosa.stft
    # --- 0.023 s. by default (using hop_length = win_length // 4, win_length = n_fft,  n_fft = 2048 by default and sr_hz = 22050)
    step_sec = time_sec_v[1] - time_sec_v[0] 
    pos_frame_v = np.arange(0, data_m.shape[1], int(np.floor(step_target_sec / step_sec)))
    # --- https://librosa.org/doc/main/generated/librosa.util.sync.html
    data_sync_m = librosa.util.sync(data=data_m, idx=pos_frame_v, aggregate=np.mean, pad=False)
    time_sync_sec_v = 0.5*(time_sec_v[pos_frame_v[0:-1]]+time_sec_v[pos_frame_v[1:]])
    return data_sync_m, time_sync_sec_v


def f_patches(data_m: np.ndarray,
            time_sec_v: np.ndarray,
            patch_halfduration_frame: int = 20,
            patch_hop_frame: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert data_m to a list of patches of length 2*patch_halfduration_frame

    Args:
        data (dim, nb_frame)
        time_sec_v (nb_frame,)

    Returns:
        data (nb_patch, dim, 2*patch_halfduration_frame)
        time (nb_patch,)
    """

    middle_frame = patch_halfduration_frame
    nb_frame = data_m.shape[1]
    data_l = []
    time_sec_l = []
    while middle_frame + patch_halfduration_frame < nb_frame:
        data_l.append(data_m[:, middle_frame - patch_halfduration_frame:
                                middle_frame + patch_halfduration_frame])
        time_sec_l.append( time_sec_v[middle_frame ] )
        middle_frame += patch_hop_frame
    return np.asarray(data_l), np.asarray(time_sec_l)


def f_groundtruth_from_annotation(time_sec_v: np.ndarray, 
                                  annot_l: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct a ground-truth Self-Similarity-Matrix (SSM) and novelvely curve from annotations

    Args:
        time_sec_v (nb_frame,): target time axis of the SSM
        annot_l:    list of structure segments (each segment if a dictionary with key 'time', 'duration', 'value')
    Returns:
        gt_SSM_m (nb_frame,): 
        gt_novelty_v (nb_frame,): 
    """

    # --- get the total number of class
    dict_label_l = list(set([seg['value'] for seg in annot_l]))
    nb_class = len(dict_label_l)
    
    # --- for segment, check A < time_sec_v < B, assign the correponding class    
    label_state_m = np.zeros((nb_class, len(time_sec_v)))
    label_state_v = np.zeros((len(time_sec_v)))
    #label_seq_m = np.zeros((nb_class*1000, len(time_sec_v)))
    for seg in annot_l:
        class_idx = dict_label_l.index(seg['value'])
        pos_v = np.where( (seg['time'] < time_sec_v) & (time_sec_v <= seg['time']+seg['duration']) )[0]
        # --- state
        label_state_m[class_idx, pos_v] = 1
        label_state_v[pos_v] = class_idx
        # --- sequence
        #for idx, pos in enumerate(pos_v):
        #    label_seq_m[(class_idx*1000)+idx, pos:pos+3] = 1
    gt_SSM_m = 1.0*label_state_m.T.dot(label_state_m)
    #gt_SSM_m += 0.3*label_seq_m.T.dot(label_seq_m)

    pos_v = np.where(np.diff(label_state_v)!=0)[0]+1
    boundary_v = np.zeros((len(time_sec_v)))
    boundary_v[pos_v] = 1
    gt_novelty_v = convolve(boundary_v, np.array([0.25, 0.5, 1, 0.5, 0.25]), 'same')

    return gt_SSM_m, gt_novelty_v
