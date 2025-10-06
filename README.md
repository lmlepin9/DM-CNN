# DM-CNN: Dark Matter - Liquid Argon Interaction Classifier

Convolutional neural network aimed to discriminate dark matter interactions with liquid argon (dark trident scattering) from 
neutrino interactions and from cosmic-ray muons. DM-CNN repurposes the MPID package to 
train a binary classifier. Analogously to MPID, this network receives 512x512 LArTPC images. The CNN returns the probability
of the image containing either a dark trident interaction or a background interaction. 


<img src="https://github.com/lmlepin9/DM-CNN/blob/master/lib/run1_NuMI_beamon_larcv_cropped_ENTRY_4204_colorbar_logit.png" width="500">

## Dependecies
[LArCV2 v2.0.0](https://github.com/DeepLearnPhysics/larcv2),
ROOT,
PyTorch

## Docker container

MPID was originally built using python 2.7 and PyTorch (V1.0.1). I have made an updated version with Python3 and CUDA 11.0 which should work on any modern GPU (hopefully!). A container with all the dependencies can be found here:

[LArCV-Py3 Container](https://hub.docker.com/repository/docker/lmlepin9/larcv2_py3/general)

## Setup
0. Download the container 
1. Clone this repo 
2. Activate the docker container
3. Setup dependencies and MPID core: source setup_larcv2_dm.sh 

## Training
0. Declare training parameters and paths in ./cfg/training_config.cfg 
2. python ./uboone/train_DM-CNN.py 

Note: You might want to run this command with nohup, so the training will continue even if you close your terminal 

## Inference
0. Declare paths in ./cfg/inference_config_binary.cfg 
1. python ./uboone/inference_DM-CNN.py

## Event display

We can also use the LArCV tools to create event displays 
of the images we feed to the CNN. 

0. Declare display parameters and paths in ./cfg/print_image_config.cfg
1. python ./uboone/print_image_with_score.py -n entry_number

## Occlusion analysis

A key part of performing a HEP analysis using deep learning tools is to understand
what features are meaningful for the DL model. One strategy to evaluate what pixels
are important to the CNN is to use a occlusion analysis (see [arXiv:1311.2901](https://arxiv.org/abs/1311.2901)). In this repo we include
an script to perform an occlusion analysis over the larcv input images. 

0. Declare paths and occlusion box size in ./cfg/occlusion_config.cfg 
1. python ./uboone/occlusion_analysis_CNN.py -n entry_number 

