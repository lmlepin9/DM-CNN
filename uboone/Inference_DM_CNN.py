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
from mpid_net import mpid_net_binary, mpid_func

plt.ioff()
torch.cuda.is_available()

from lib.config import config_loader
MPID_PATH = os.path.dirname(mpid_data_binary.__file__)+"/../cfg"
CFG = os.path.join(MPID_PATH,"inference_config_binary.cfg")

# CFG = os.path.join("../../cfg","inference_config.cfg")
cfg  = config_loader(CFG)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=cfg.GPUID


#input_channel = "dt_overlay_"+mass
#input_file = "dt_overlay_"+mass+"_larcv_cropped.root"
#output_dir = "/hepgpu4-data2/lmlepin/outputs/"
#output_file = output_dir + input_channel + "_MPID_scores_true_vertex_8441_steps.csv"

input_file = "/hepgpu4-data2/lmlepin/datasets/standard_NuMI_run1_overlay_larcv_cropped.root"
weight_file="/hepgpu4-data2/lmlepin/CNN_weights/binary_cosmics_FULL_weights/mpid_model_COSMICS_FULL_20210819-08_34_PM_epoch_4_batch_id_581_labels_2_title_0.001_AG_GN_final_2_classes_step_8441.pwf"
train_device = 'cuda' if torch.cuda.is_available() else 'cpu'

mpid = mpid_net_binary.MPID()
mpid.cuda()
mpid.load_state_dict(torch.load(weight_file, map_location=train_device))
mpid.eval()



input_data = mpid_data_binary.MPID_Dataset(input_file, "image2d_image2d_binary_tree", train_device, plane=0, augment=cfg.augment)
# Training and test data
train_size = 13000
test_size = 1154
#train_size = int(0.9 * len(input_data))
#test_size = int(0.1 * len(input_data))  
train_data, test_data = torch.utils.data.random_split(input_data,[train_size, test_size])

print(np.shape(test_data)) 

electron_score = []
gamma_score = []
muon_score = []
electron_label = []
gamma_label = []
muon_label = []

print("Total number of events: ", test_size)

print("Starting...")



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

                                            
pre_df_dict = {'e_score':electron_score,'gamma_score':gamma_score,'mu_score':muon_score,
                'e_label':electron_label, 'gamma_label':gamma_label, 'mu_label':muon_label}
df = pd.DataFrame.from_dict(pre_df_dict)
df.to_csv(output_file,index=False)
