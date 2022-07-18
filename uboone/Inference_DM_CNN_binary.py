import os, sys, ROOT
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataloader as dataloader

from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from mpid_data import mpid_data_binary
from mpid_net import mpid_net_binary, mpid_func_binary

plt.ioff()
torch.cuda.is_available()
print(torch.cuda.is_available())

from lib.config import config_loader
MPID_PATH = os.path.dirname(mpid_data_binary.__file__)+"/../cfg"
CFG = os.path.join(MPID_PATH,"inference_config.cfg")
CFG = "/hepgpu5-data1/yuliia/MPID/DM-CNN/cfg/inference_config.cfg"

cfg  = config_loader(CFG)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=cfg.GPUID

output_dir = "/hepgpu5-data1/yuliia/training_output/"
output_file = output_dir + "MPID_scores_mpid_cosmics_mc_test_9291_steps.csv"

#input_csv = pd.read_csv("/hepgpu5-data1/yuliia/spg_photon.csv") # event info, vertices, energy
input_file = "/hepgpu5-data1/yuliia/MPID/larcv2/mpid_cosmics_mc_test.root"  # cropped images

weight_file="/hepgpu5-data1/yuliia/MPID_pytorch/weights/mpid_model_20220716-08_46_PM_epoch_5_batch_id_1541_labels_2_title_0.001_AG_GN_final_step_9291.pwf"

# Set up GPU
train_device = 'cuda' if torch.cuda.is_available() else 'cpu'

mpid = mpid_net_binary.MPID()
mpid.cuda()
mpid.load_state_dict(torch.load(weight_file, map_location=train_device))
mpid.eval()

input_data = mpid_data_binary.MPID_Dataset(input_file, "image2d_image2d_binary_tree", train_device, plane=0, augment=False)

signal_score = []
bkg_score = []

run = []
subrun = []
event = []

print("Starting...")

n_events = len(input_data)

print("Total number of events: ", n_events)

for ENTRY in range(n_events - 1):
    if(ENTRY%1000 == 0):
        print("ENTRY: ", ENTRY)

    input_image = input_data[ENTRY][0].view(-1,1,512,512)

    signal_label = input_data[ENTRY][1][0].item()

    input_image[0][0][input_image[0][0] > 500] = 500
    input_image[0][0][input_image[0][0] < 10 ] = 0

    score = nn.Sigmoid()(mpid(input_image.cuda())).cpu().detach().numpy()[0]
    signal_score.append(score[0])
    bkg_score.append(score[1])

    run.append(input_data[ENTRY][2][0].item())
    subrun.append(input_data[ENTRY][2][1].item())
    event.append(input_data[ENTRY][2][2].item())
                                  
pre_df_dict = {'run': run, 'subrun': subrun, 'event': event, 'signal_label': signal_label, 'signal_score': signal_score, 'bkg_score': bkg_score}
df = pd.DataFrame.from_dict(pre_df_dict)
df.to_csv(output_file,index=False)
