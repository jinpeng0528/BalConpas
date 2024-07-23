# Strike a Balance in Continual Panoptic Segmentation

This is an official implementation of the paper "Strike a Balance in Continual Panoptic Segmentation", submitted to ECCV 2024.

## Clarification of Anonymity
Our code is built upon some previous open-source libraries, including [Mask2Former](https://github.com/facebookresearch/Mask2Former), [Detectron2](https://github.com/facebookresearch/detectron2/tree/main), and others. These libraries originally contained some names of authors or institutions. We have made every effort to remove these. However, to ensure full coverage in case we missed anything, we hereby clarify: any names of people or institutions that appear in the code do not imply any specific association with the authors of this submission.

Additionally, subsequent sections may refer to the installation of certain packages or the downloading of datasets. These references are purely for sourcing purposes and should not be interpreted as having any relation to the authors of this submission.

## Installation

### Environment
To install the required environment (for CUDA 11.3), run:
```bash
conda create -n balconpas python=3.8.17
conda activate balconpas
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
pip install git+https://github.com/cocodataset/panopticapi.git
pip install -r requirements.txt
```

### CUDA kernel for MSDeformAttn
After preparing the required environment, run the following command to compile CUDA kernel for MSDeformAttn:

`CUDA_HOME` must be defined and points to the directory of the installed CUDA toolkit.

```bash
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```

[//]: # (### Pre-trained ResNet-101)

[//]: # (For continual panoptic segmentation and continual instance segmentation, we use the ResNet-50 backbone, whose pre-trained model can be automatically downloaded by Detectron2. )

[//]: # (However, for continual semantic segmentation, we use the ResNet-101 backbone, whose pre-trained model needs to be manually downloaded from [here]&#40;https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-101.pkl&#41; and placed in the `./` directory.)


## Data Preparation

Please download the ADE20K dataset and its instance annotation from [here](http://sceneparsing.csail.mit.edu/), then place the dataset in or create a symbolic link to the `./datasets` directory. The structure of data path should be organized as follows:
```
ADEChallengeData2016/
  images/
  annotations/
  objectInfo150.txt
  sceneCategories.txt
  annotations_instance/
  annotations_detectron2/
  ade20k_panoptic_{train,val}.json
  ade20k_panoptic_{train,val}/
  ade20k_instance_{train,val}.json
```
The directory `annotations_detectron2` is generated by running `python datasets/prepare_ade20k_sem_seg.py`.
Then, run `python datasets/prepare_ade20k_pan_seg.py` to combine semantic and instance annotations for panoptic annotations and run `python datasets/prepare_ade20k_ins_seg.py` to extract instance annotations in COCO format.

To fit the requirements of continual segmentation tasks, run `python continual/prepare_datasets.py` to reorganize the annotations (reorganized annotations will be placed in `./json`).

To generate replay samples for each step, run `python continual/memory_generator/memory_selection_pan.py`, `python continual/memory_generator/memory_selection_sem.py`, and `python continual/memory_generator/memory_selection_inst.py` for panoptic, semantic, and instance segmentation, respectively (replay samples will be placed in `./json/memory`).

#### Example data preparation
```bash
# for Mask2Former
cd datasets
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
unzip ADEChallengeData2016.zip
cd ADEChallengeData2016
wget http://sceneparsing.csail.mit.edu/data/ChallengeData2017/annotations_instance.tar
tar -xvf annotations_instance.tar
cd ../..
python datasets/prepare_ade20k_sem_seg.py
python datasets/prepare_ade20k_pan_seg.py
python datasets/prepare_ade20k_ins_seg.py

# for continual segmentation
python continual/prepare_datasets.py

# for memory
python continual/memory_generator/memory_selection_pan.py
python continual/memory_generator/memory_selection_sem.py
python continual/memory_generator/memory_selection_inst.py
```

## Getting started

### Training
Herein, we provide an example script for training the 100-10 continual panoptic segmentation of the ADE20K dataset:
```bash
python train_continual.py --config-file configs/ade20k/panoptic-segmentation/100-10.yaml \
CONT.TASK 1 SOLVER.BASE_LR 0.0001 SOLVER.MAX_ITER 160000 OUTPUT_DIR ./output/ps/100-10/step1

for t in 2 3 4 5 6; do
  python train_continual.py --config-file configs/ade20k/panoptic-segmentation/100-10.yaml \
  CONT.TASK ${t} SOLVER.BASE_LR 0.00005 SOLVER.MAX_ITER 10000 OUTPUT_DIR ./output/ps/100-10/step${t}
done
```

For more training scripts, please refer to the `./scripts` directory.

### Evaluation
To evaluate the trained model, add the argument `--eval-only --resume` to the command line. For example:
```bash
python train_continual.py --eval-only --resume --config-file configs/ade20k/panoptic-segmentation/100-10.yaml \
CONT.TASK 6 OUTPUT_DIR ./output/ps/100-10/step6
```