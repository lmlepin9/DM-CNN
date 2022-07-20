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

from mpid_data import mpid_data
from mpid_net import mpid_net, mpid_func

plt.ioff()
torch.cuda.is_available()
print(torch.cuda.is_available())

from lib.config import config_loader
MPID_PATH = os.path.dirname(mpid_data.__file__)+"/../cfg"
CFG = os.path.join(MPID_PATH,"inference_config.cfg")
CFG = "/hepgpu5-data1/yuliia/MPID/DM-CNN/cfg/inference_config.cfg"

# CFG = os.path.join("../../cfg","inference_config.cfg")
cfg  = config_loader(CFG)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=cfg.GPUID

#input_channel = "dt_overlay_"+mass
#input_file = "dt_overlay_"+mass+"_larcv_cropped.root"

output_dir = "/hepgpu5-data1/yuliia/training_output/"
output_file = output_dir + "MPID_scores_mpid_cosmics_mc_test_9291_steps.csv"

#input_csv = pd.read_csv("/hepgpu5-data1/yuliia/spg_photon.csv") # event info, vertices, energy
input_file = "/hepgpu5-data1/yuliia/MPID/larcv2/mpid_cosmics_mc_test.root"  # cropped images

weight_file="/hepgpu5-data1/yuliia/MPID_pytorch/weights/mpid_model_20220716-08_46_PM_epoch_5_batch_id_1541_labels_2_title_0.001_AG_GN_final_step_9291.pwf"

# Set up GPU
train_device = 'cuda' if torch.cuda.is_available() else 'cpu'

mpid = mpid_net.MPID()
mpid.cuda()
mpid.load_state_dict(torch.load(weight_file, map_location=train_device))
mpid.eval()

input_data = mpid_data.MPID_Dataset(input_file, "image2d_image2d_tree", train_device, nclasses=3, plane=0, augment=cfg.augment)

# If training and test data are not separate
"""train_size = int(0.9 * len(input_data))
#test_size = int(0.1 * len(input_data))  
#train_data, test_data = torch.utils.data.random_split(input_data,[train_size, test_size])"""

electron_score = []
gamma_score = []
muon_score = []
electron_label = []
gamma_label = []
muon_label = []

run = []
subrun = []
event = []
energy = []
momentum = []

print("Starting...")

test_data = input_data
test_size = len(test_data)
n_events = test_size

print("Total number of events: ", n_events)

for ENTRY in range(n_events - 1):
    if(ENTRY%1000 == 0):
        print("ENTRY: ", ENTRY)

    input_image = test_data[ENTRY][0].view(-1,1,512,512)

    true_label = test_data[ENTRY][1]
    electron_label.append(true_label[0])
    gamma_label.append(true_label[1])
    muon_label.append(true_label[2])

    input_image[0][0][input_image[0][0] > 500] = 500
    input_image[0][0][input_image[0][0] < 10 ] = 0

    score = nn.Sigmoid()(mpid(input_image.cuda())).cpu().detach().numpy()[0]
    electron_score.append(score[0])
    gamma_score.append(score[1])
    muon_score.append(score[2])

    run.append(test_data[ENTRY][2][0])
    subrun.append(test_data[ENTRY][2][1])
    event.append(test_data[ENTRY][2][2])

    # Search the full input data by run & event 
    energy_value = input_csv.loc[(input_csv['run'] == run[-1]) & (input_csv['event'] == event[-1]), 'Energy'].values[0]
    energy.append(energy_value)
    momentum_value = input_csv.loc[(input_csv['run'] == run[-1]) & (input_csv['event'] == event[-1]), 'Momentum'].values[0]
    momentum.append(momentum_value)
                                  
pre_df_dict = {'run': run, 'e_score':electron_score,'gamma_score':gamma_score,'mu_score':muon_score,
               'subrun': subrun, 'e_label':electron_label, 'gamma_label':gamma_label, 'mu_label':muon_label,
               'event': event, 'Energy': energy, 'Momentum': momentum}
df = pd.DataFrame.from_dict(pre_df_dict)
df.to_csv(output_file,index=False)
