# SSM-Net

Official implementation of [Self-Similarity-Based and Novelty-based loss for music structure analysis](https://arxiv.org/pdf/2309.02243.pdf), published at [ISMIR 2023](https://ismir2023.ismir.net/).
Include a pre-trained model.


## Citation

If you use this code and/or paper in your research please cite:

```
@inproceedings{SSMNet,
    author = {Peeters, Geoffroy},
    booktitle = {Proceedings of the 24th International Society for Music Information Retrieval Conference, ISMIR 2023},
    publisher = {International Society for Music Information Retrieval},
    title = {Self-Similarity-Based and Novelty-based loss for music structure analysis},
    year = {2023}
}
```

## Installation

```
git clone https://github.com/geoffroypeeters/ssmnet_ISMIR2023.git
cd ssmnet_ISMIR2023/

python -m venv env_ssmnet
source env_ssmnet/bin/activate

pip install -e .
``````

## Usage

### Command-line interface

This package includes a CLI as well as pretrained models. To use it, type in a terminal:
```
ssmnet $fullpath_to_audio_file -o csv_file -p pdf_file
```


### Output formats

The output format is .csv. The output file is specified with -o.

```
segment_start_time_sec,segment_stop_time_sec,segment_label
0.000,11.192,1.000
11.192,25.588,1.000
25.588,37.663,1.000
37.663,50.666,1.000
...
```

### Python API

Alternatively, the functions defined in `ssmnet/core.py` can directly be called within another Python code.
```python
ssmnet_deploy = SsmNetDeploy(config_d)

# get the audio features patches
feat_3m, time_sec_v = ssmnet_deploy.m_get_features(args.audio_file)
# process through SSMNet to get the Self-Similarity-Matrix and Novelty-Curve
hat_ssm_np, hat_novelty_np = ssmnet_deploy.m_get_ssm_novelty(feat_3m)
# estimate segment boundries from the Novelty-Curve
hat_boundary_sec_v, hat_boundary_frame_v = ssmnet_deploy.m_get_boundaries(hat_novelty_np, time_sec_v)

# export as .csv
ssmnet_deploy.m_plot(hat_ssm_np, hat_novelty_np, hat_boundary_frame_v, args.output_pdf_file)
# export as .pdf
ssmnet_deploy.m_export_csv(hat_boundary_sec_v, args.output_csv_file)
```



## Code organization

```
groundtruth
    |--harmonix.pyjama
    |--isophonics.pyjama
    |--rwc-pop.pyjama
    |--salami.pyjama
```
contains the pyjama file (see [doc](https://github.com/geoffroypeeters/pyjama) for a description of the pyjama format) corresponding to the four datasets. Each pyjama file contains all the annotations of a given dataset.

Those are provided for reproducibility.
In the paper, the evaluation using
- `salami-pop` is performed using the subset of entries of `salami.pyjama` with key `CLASS` equal to `popular`
- `salami-ia` is done using the subset of entries of `salami.pyjama` with key `SOURCE` equal to `IA`
- `salami-two` is done using the subset of entries of `salami.pyjama` which has two annotations (both `textfile1_functions.txt` and `textfile2_functions.txt` keys are defined)

```python
with open('salami.pyjama', encoding = "utf-8") as json_fid: data_d = json.load(json_fid)

subentry_l = [entry for entry in data_d['collection']['entry'] if entry['CLASS'][0]['value']=='popular']
len(subentry_l) # ---> 280

subentry_l = [entry for entry in data_d['collection']['entry'] if entry['SOURCE'][0]['value']=='IA']
len(subentry_l) # ---> 446

subentry_l = [entry for entry in data_d['collection']['entry'] if len(entry['textfile1_functions.txt']) and len(entry['textfile2_functions.txt'])]
len(subentry_l) # ---> 882
```

```
ssmnet
    |--/weights_deploy/
            |--*.pt  # weights of pre-trained network for the model with `do_nb_attention=1` and `do_nb_attention=3`
    |--core
    |--example
    |--model.py # --- is the pytorch code of the model
    |--utils.py
```

contains the code of the library

