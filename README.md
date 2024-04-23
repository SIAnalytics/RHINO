# Hausdorff Distance Matching with Adaptive Query Denoising for Rotated Detection Transformer
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rhino-rotated-detr-with-dynamic-denoising-via/oriented-object-detction-on-dota-2-0)](https://paperswithcode.com/sota/oriented-object-detction-on-dota-2-0?p=rhino-rotated-detr-with-dynamic-denoising-via)

**[SI Analytics](https://www.si-analytics.ai/)**

[Hakjin Lee](https://github.com/nijkah), Minki Song, Jamyoung Koo, [Junghoon Seo](https://scholar.google.co.kr/citations?user=9KBQk-YAAAAJ)

[[`Paper`](https://arxiv.org/abs/2305.07598)]

The RHINO is a robust DETR architecture designed for detecting rotated objects. It demonstrates promising results, exceeding 60 mAP on DOTA-v2.0.

## Main Results
DOTA-v2.0 (Single-Scale Training and Testing)
| Method |         Backbone         | AP50  |                            Config                          | Download |
| :----: | :----------------------: | :---: | :----------------------------------------------------------: |  :----: |
|RHINO| ResNet50 (1024,1024,200) | 59.26 |    [rhino_r50_dota2](configs/rhino/rhino_phc_haus-4scale_r50_2xb2-36e_dotav2.py)      |  [model]() |
|RHINO| Swin-T (1024,1024,200) | 60.72 |     [rhino_swint_dota2](configs/rhino/rhino_phc_haus-4scale_swint_2xb2-36e_dotav2.py)      | [model]() |

DOTA-v1.5 (Single-Scale Training and Testing)
| Method |         Backbone         | AP50  |                            Config                          | Download |
| :----: | :----------------------: | :---: | :----------------------------------------------------------: |  :----: |
|RHINO| ResNet50 (1024,1024,200) | 71.96 |    [rhino_r50_dotav15](configs/rhino/rhino_phc_haus-4scale_r50_2xb2-36e_dotav15.py)      |  [model]() |
|RHINO| Swin-T (1024,1024,200) | 73.46 |     [rhino_swint_dotav15](configs/rhino/rhino_phc_haus-4scale_swint_2xb2-36e_dotav15.py)      | [model]() |

DOTA-v1.0 (Single-Scale Training and Testing)
| Method |         Backbone         | AP50  |                            Config                          | Download |
| :----: | :----------------------: | :---: | :----------------------------------------------------------: |  :----: |
|RHINO| ResNet50 (1024,1024,200) | 78.68 |    [rhino_r50_dota](configs/rhino/rhino_phc_haus-4scale_r50_2xb2-36e_dota.py)      |  [model]() |
|RHINO| Swin-T (1024,1024,200) | 79.42 |     [rhino_swint_dota](configs/rhino/rhino_phc_haus-4scale_swint_2xb2-36e_dota.py)      | [model]() |


## Requirements

### Installation
```bash
# torch>=1.9.1 is required.
pip install openmim mmengine==0.7.3
mim install mmcv==2.0.0
pip install mmdet==3.0.0
pip3 install --no-cache-dir -e ".[optional]"
```
or check the [Dockerfile](docker/Dockerfile).


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
# DOTA-v2.0 R-50
export CONFIG='configs/rhino/rhino_phc_haus-4scale_r50_2xb2-36e_dotav2.py'
bash tools/dist_train.sh $CONFIG 2
```

## Evaluation

To evaluate our models on DOTA, run:

```bash
# example
export CONFIG='configs/rhino/rhino_phc_haus-4scale_r50_2xb2-36e_dotav2.py'
export CKPT='work_dirs/rhino_phc_haus-4scale_r50_2xb2-36e_dotav2/epoch_36.pth'
python3 tools/test.py $CONFIG $CKPT
```
Evaluation is processed in the [official DOTA evaluation server](https://captain-whu.github.io/DOTA/evaluation.html).


## License
This project is licensed under CC-BY-NC. It is available for academic purposes only.


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
