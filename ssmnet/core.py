from __future__ import annotations

import os
import sys
import yaml
import pdb
import pprint as pp
import numpy as np
from typing import Tuple

import warnings
try:
    MATPLOTLIB_AVAILABLE = True
    import matplotlib.pyplot as plt
except (IndexError, ModuleNotFoundError):
    MATPLOTLIB_AVAILABLE = False


#import matplotlib.pyplot as plt
import librosa
import torch

import ssmnet.utils
import ssmnet.model



class SsmNetDeploy():

    def __init__(self, config_d: dict):
        """
        Args:
            dictionary coming from configuration file
        """
        self.config_d = config_d
        return


    def m_get_features(self, 
                       audio_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the audio features

        Args:
            audio_file
        Returns:
            feat_3m, 
            time_sec_v
        """
        try:
            audio_v, sr_hz = librosa.load(audio_file)
        except:
            sys.exit(f'something wrong in reading audio file "{audio_file}"')
        
        if len(audio_v)==0:
            sys.exit(f'something wrong in reading audio file "{audio_file}"')

        logmel_m, time_sec_v = ssmnet.utils.f_extract_feature(audio_v, sr_hz)
        logmel_sync_m, time_sync_sec_v = ssmnet.utils.f_reduce_time(
            logmel_m,
            time_sec_v,
            self.config_d['features']['step_target_sec'])
        feat_3m, time_sec_v = ssmnet.utils.f_patches(
            logmel_sync_m, time_sync_sec_v,
            self.config_d['features']['patch_halfduration_frame'],
            self.config_d['features']['patch_hop_frame'])

        self.step_sec = time_sec_v[1]-time_sec_v[0]
    
        return feat_3m, time_sec_v
    

    def m_get_ssm_novelty(self,
                          feat_3m: np.ndarray)  -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the Self-Similarity-Matrix and novelty-curve using a pre-trained SSM-Net

        Args:
            feat_3m
        Returns:
            hat_ssm_np
            hat_novelty_np
        """
        # --- using torchlightning
        #import ssm_lightning
        #my_lighting = ssm_lightning.SsmLigthing.load_from_checkpoint(config_d['model']['file'])
        #hat_novelty_v, hat_ssm_m = my_lighting.model.get_novelty( torch.from_numpy(feat_3m) )
        # --- using ony torch
        model = ssmnet.model.SsmNet(self.config_d['model'], self.step_sec)
        file_state_dict =  os.path.join(os.path.dirname(__file__), 
                                        "weights_deploy", 
                                        self.config_d['model']['file'].replace('.ckpt', '_state_dict.pt'))
        if False: # --- load torchlightning -> need to convert, depends on the exact path of modules :-()
            data = torch.load(self.config_d['model']['file'], map_location=torch.device('cpu'))
            data_clean = {}
            for key in data['state_dict'].keys(): 
                data_clean[key.replace('model.', '')] = data['state_dict'][key]
            model.load_state_dict(data_clean)
            # --- export to only state_dict -> this is the way to go
            torch.save(model.state_dict(), file_state_dict)
        else:
            model.load_state_dict(torch.load(file_state_dict))
        
        hat_novelty_v, hat_ssm_m = model.get_novelty( torch.from_numpy(feat_3m) )
        hat_novelty_np = hat_novelty_v.detach().squeeze().numpy()
        hat_ssm_np = hat_ssm_m.detach().squeeze().numpy()

        return hat_ssm_np, hat_novelty_np


    def m_get_boundaries(self, 
                         hat_novelty_np: np.ndarray, 
                         time_sec_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate the boundaries

        Args:
            hat_novelty_np
            time_sec_v
        Returns:
            hat_boundary_sec_v, 
            hat_boundary_frame_v
        """
        hat_boundary_frame_v = ssmnet.utils.f_get_peaks(hat_novelty_np, self.config_d['postprocessing'], self.step_sec)

        hat_boundary_sec_v = time_sec_v[hat_boundary_frame_v]
        # --- add start and end-time
        hat_boundary_sec_v = np.concatenate((0*np.ones(1), hat_boundary_sec_v, time_sec_v[-1]*np.ones(1)))
        # --- to be sure there is not twice zero
        hat_boundary_sec_v = sorted([aaa for aaa in set(hat_boundary_sec_v)])

        return hat_boundary_sec_v, hat_boundary_frame_v


    def m_plot(self, 
               hat_ssm_np: np.ndarray, 
               hat_novelty_np: np.ndarray, 
               hat_boundary_frame_v: np.ndarray,
               output_file: str):
        """
        Plot and save to pdf file

        Args:
            hat_ssm_np
            hat_novelty_np
            hat_boundary_frame_v
            output_file
        Returns:

        """
        if not MATPLOTLIB_AVAILABLE:
            warnings.warn("Exporting in pdf format requires Matplotlib to be installed. Skipping...")
            return

        plt.clf()
        plt.imshow(hat_ssm_np)
        plt.colorbar()
        nb_frame = hat_ssm_np.shape[0]
        plt.plot((1-hat_novelty_np/max(hat_novelty_np))*nb_frame, 'r', linewidth=1)
        for x in hat_boundary_frame_v:
            plt.plot([x,x], [nb_frame,0], 'm', linewidth=1)
        plt.savefig(output_file)

        return
    
    def m_export_csv(self, 
                hat_boundary_sec_v: np.ndarray, 
                output_file: str):
        """
        Export boundary in a .csv file

        Args:
            hat_boundary_sec_v
            output_file
        Returns:

        """
        start = hat_boundary_sec_v[0:-1]
        stop = hat_boundary_sec_v[1:]
        label = np.ones(len(start)) 
        data = np.stack((start, stop, label), axis=1)
        header = "segment_start_time_sec,segment_stop_time_sec,segment_label"

        np.savetxt(output_file, data, delimiter=',', fmt='%.3f', header=header, comments="")


        return