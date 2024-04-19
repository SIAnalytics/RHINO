# Hausdorff Distance Matching with Adaptive Query Denoising for Rotated Detection Transformer

- This repository is the official implementation of RHINO.

## Requirements

### install by pip
```bash
# torch>=1.9.1 is required.
pip install openmim mmengine==0.7.3
mim install mmcv==2.0.0
pip install mmdet==3.0.0
pip3 install --no-cache-dir -e ".[optional]"
```


### Main Results
DOTA-v2.0
| Method |         Backbone         | AP50  |                            Config                          | Download |
| :-----: | :----------------------: | :---: | :----------------------------------------------------------: |  :----: |
|RHINO| ResNet50 (1024,1024,200) | 59.26 |    [rhino_r50_dota2](configs/rhino/rhino_phc_haus-4scale_r50_8xb2-36e_dotav2.py)      |  [model]() |
|RHINO| Swin-T (1024,1024,200) | 60.72 |     [rhino_swint_dota2](configs/rhino/rhino_phc_haus-4scale_swint_8xb2-36e_dotav2.py)      | [model]() |


### Preprare Dataset
Details are described in https://github.com/open-mmlab/mmrotate/blob/main/tools/data/dota/README.md

Specifically, run below code.

```bash
python3 tools/data/dota/split/img_split.py --base-json \
  tools/data/dota/split/split_configs/ss_trainval.json

python3 tools/data/dota/split/img_split.py --base-json \
  tools/data/dota/split/split_configs/ss_test.json
```


## Training

To train the model(s) in the paper, run this command:

```bash
# DOTA-v1.0 R-50
export CONFIG='configs/rhino/rhino_phc_haus-4scale_r50_8xb2-36e_dota.py'
# DOTA-v1.0 Swin-T
export CONFIG='configs/rhino/rhino_phc_haus-4scale_swint_8xb2-36e_dota.py'

# DOTA-v1.5 R-50
export CONFIG='configs/rhino/rhino_phc_haus-4scale_r50_8xb2-36e_dotav15.py'
# DOTA-v1.5 Swin-T
export CONFIG='configs/rhino/rhino_phc_haus-4scale_swint_8xb2-36e_dotav15.py'

# DOTA-v2.0 R-50
export CONFIG='configs/rhino/rhino_phc_haus-4scale_r50_8xb2-36e_dotav2.py'
# DOTA-v2.0 Swin-T
export CONFIG='configs/rhino/rhino_phc_haus-4scale_swint_8xb2-36e_dotav2.py'

# DIOR R-50
export CONFIG='configs/rhino/rhino_phc_haus-4scale_r50_8xb2-36e_dior.py'
# DIOR Swin-T
export CONFIG='configs/rhino/rhino_phc_haus-4scale_swint_8xb2-36e_dior.py'
bash tools/dist_train.sh $CONFIG 2
```

## Evaluation

To evaluate our models on DOTA, run:

```bash
# example
export CONFIG='configs/rhino/rhino_phc_haus-4scale_r50_8xb2-36e_dota.py'
export CKPT='work_dirs/rhino_phc_haus-4scale_r50_8xb2-36e_dota/epoch_36.pth'
python3 tools/test.py $CONFIG $CKPT
```
Evaluation is processed in the [official DOTA evaluation server](https://captain-whu.github.io/DOTA/evaluation.html).



# Bibtex
If you find our work helpful for your research, please consider citing the following BibTeX entry.
```bibtex
@misc{lee2023hausdorff,
      title={Hausdorff Distance Matching with Adaptive Query Denoising for Rotated Detection Transformer},
      author={Hakjin Lee and Minki Song and Jamyoung Koo and Junghoon Seo},
      year={2023},
      eprint={2305.07598},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
