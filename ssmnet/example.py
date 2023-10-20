# python -m ssmnet_example -c ./config_example.yaml -a /home/ids/gpeeters/M2-ATIAM-internship/music-structure-estimation/_references/rwc-pop/audio/RM-P001.wav

from .parser import parse_args
from .core import SsmNetDeploy
import yaml
import pdb
import os

def ssmnet_main():
    """

    """
    args = parse_args()
    args.config_file =  os.path.join(os.path.dirname(__file__), "weights_deploy", args.config_file)
    print(f'parsing yaml file "{args.config_file}"')
    with open(args.config_file, "r", encoding="utf-8") as fid:
        config_d = yaml.safe_load(fid)

    ssmnet_deploy = SsmNetDeploy(config_d)
    feat_3m, time_sec_v = ssmnet_deploy.m_get_features(args.audio_file)
    hat_ssm_np, hat_novelty_np = ssmnet_deploy.m_get_ssm_novelty(feat_3m)
    hat_boundary_sec_v, hat_boundary_frame_v = ssmnet_deploy.m_get_boundaries(hat_novelty_np, time_sec_v)
    ssmnet_deploy.m_plot(hat_ssm_np, hat_novelty_np, hat_boundary_frame_v, args.output_pdf_file)
    ssmnet_deploy.m_export_csv(hat_boundary_sec_v, args.output_csv_file)



if __name__ == "__main__":
    ssmnet_main()