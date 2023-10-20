# -*- coding: utf-8 -*-
# 2023/09/20 test deployment of ssmnet for github
# --- TODO: 
# - convert plot to second (instead of frames)
# - check package dependence
# --- USAGE:
# python -m ssmnet_example -c ./config_example.yaml -a /home/ids/gpeeters/M2-ATIAM-internship/music-structure-estimation/_references/rwc-pop/audio/RM-P001.wav

from __future__ import annotations

import sys
from argparse import ArgumentParser
import yaml
import pdb
import pprint as pp
import numpy as np

#import matplotlib.pyplot as plt
import librosa
import torch

import ssmnet.utils
import ssmnet.model

from typing import Tuple

class SsmNetDeploy():

    def __init__(self, config_d: dict):
        """
        """
        print('__init__')

        self.config_d = config_d
        return


    def m_get_features(self, 
                       audio_file: str) -> Tuple[np.ndarray, np.ndarray]:
    #def m_get_features(self, audio_file):
        """
        Compute the audio features

        Args:
            audio_file
        Returns:
            feat_3m, 
            time_sec_v
        """
        print('m_get_features')

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
        print('m_get_ssm_novelty')

        # --- using torchlightning
        #import ssm_lightning
        #my_lighting = ssm_lightning.SsmLigthing.load_from_checkpoint(config_d['model']['file'])
        #hat_novelty_v, hat_ssm_m = my_lighting.model.get_novelty( torch.from_numpy(feat_3m) )
        # --- using ony torch
        model = ssmnet.model.SsmNet(self.config_d['model'], self.step_sec)
        file_state_dict =  self.config_d['model']['file'].replace('.ckpt', '_state_dict.pt')
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
        print('m_get_boundaries')

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
               hat_boundary_frame_v: np.ndarray):
        """
        Plot and save to pdf file

        Args:
            hat_ssm_np
            hat_novelty_np
            hat_boundary_frame_v
        Returns:

        """
        print('m_plot')

        plt.clf()
        plt.imshow(hat_ssm_np)
        plt.colorbar()
        nb_frame = hat_ssm_np.shape[0]
        plt.plot((1-hat_novelty_np/max(hat_novelty_np))*nb_frame, 'r', linewidth=1)
        for x in hat_boundary_frame_v:
            plt.plot([x,x], [nb_frame,0], 'm', linewidth=1)
        plt.savefig('./fig_deploy.pdf')

        return
    



if __name__ == "__main__":

    parser = ArgumentParser(description='Compute SSM, novelty and boundaries using SSM-Net')
    parser.add_argument("-a", "--audio_file", 
                        help='audio file to process')
    parser.add_argument("-o", "--output_file", 
                        help='output file that contains the boundary positions [in sec]')
    parser.add_argument("-c", "--config_file", 
                        help='fullpath to a yaml configuration file', required=True)
    args = parser.parse_args()

    print(f'parsing yaml file {args.config_file}')
    with open(args.config_file, "r", encoding="utf-8") as fid: config_d = yaml.safe_load(fid)

    ssmnet_deploy = SsmNetDeploy(config_d)
    feat_3m, time_sec_v = ssmnet_deploy.m_get_features(args.audio_file)
    hat_ssm_np, hat_novelty_np = ssmnet_deploy.m_get_ssm_novelty(feat_3m)
    hat_boundary_sec_v, hat_boundary_frame_v = ssmnet_deploy.m_get_boundaries(hat_novelty_np, time_sec_v)
    #ssmnet_deploy.m_plot(hat_ssm_np, hat_novelty_np, hat_boundary_frame_v)

    print(hat_boundary_sec_v)