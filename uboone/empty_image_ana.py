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
CFG = "/hepgpu4-data1/yuliia/MPID_repo/DM-CNN/cfg/inference_config.cfg"

cfg  = config_loader(CFG)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=cfg.GPUID

output_dir = "/hepgpu4-data1/yuliia/training_output/"
output_file = output_dir + "MPID_scores_empty_images.csv"

weight_file="/hepgpu4-data1/yuliia/MPID_pytorch/weights/mpid_model_20220628-04_43_PM_epoch_4_batch_id_1241_title_0.001_AG_GN_final_step_6625.pwf"

# Set up GPU
train_device = 'cuda' if torch.cuda.is_available() else 'cpu'

mpid = mpid_net.MPID()
mpid.cuda()
mpid.load_state_dict(torch.load(weight_file, map_location=train_device))
mpid.eval()

electron_score = []
gamma_score = []
muon_score = []
electron_label = []
gamma_label = []
muon_label = []

run = []
subrun = []
event = []

print("Starting...")

n_events = 1000

print("Total number of events: ", n_events)

for ENTRY in range(n_events):
    if(ENTRY%100 == 0):
        print("ENTRY: ", ENTRY)

    input_image = torch.zeros([512,512]).view(-1,1,512,512)

    score = nn.Sigmoid()(mpid(input_image.cuda())).cpu().detach().numpy()[0]
    electron_score.append(score[0])
    gamma_score.append(score[1])
    muon_score.append(score[2])

    run.append(ENTRY)
    subrun.append(0)
    event.append(ENTRY)
                                  
pre_df_dict = {'run': run, 'e_score':electron_score,'gamma_score':gamma_score,'mu_score':muon_score,
               'subrun': subrun, 'event': event}
df = pd.DataFrame.from_dict(pre_df_dict)
df.to_csv(output_file,index=False)
