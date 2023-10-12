# SSM-Net

Official implementation of [Self-Similarity-Based and Novelty-based loss for music structure analysis](https://arxiv.org/pdf/2309.02243.pdf), published at [ISMIR 2023](https://ismir2023.ismir.net/).
Include a pre-trained model.


## Citation

If you use this code and/or paper in your research please cite:

Geoffroy Peeters, "Self-Similarity-Based and Novelty-based loss for music structure analysis", in International Society for Music Information Retrieval Conference (ISMIR), 2023.

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

```
with open('salami.pyjama', encoding = "utf-8") as json_fid: data_d = json.load(json_fid)

subentry_l = [entry for entry in data_d['collection']['entry'] if entry['CLASS'][0]['value']=='popular']
len(subentry_l) # ---> 280

subentry_l = [entry for entry in data_d['collection']['entry'] if entry['SOURCE'][0]['value']=='IA']
len(subentry_l) # ---> 446

subentry_l = [entry for entry in data_d['collection']['entry'] if len(entry['textfile1_functions.txt']) and len(entry['textfile2_functions.txt'])]
len(subentry_l) # ---> 882
```

```
src
    |--ssm_model.py # --- is the pytorch code of the model
    |--ssm_utils.py
```

contains the code of the library

```
example
    |--ssm_example.py # --- provide an example of the code usage to extrac the structure from a given file
    |--config_example.py # --- contains the configuration of the model
    |--*.ckpt  # --- contains the checkpoint of the model for the configuration `do_nb_attention=1` and `do_nb_attention=3`
```
