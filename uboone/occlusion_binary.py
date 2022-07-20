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
import torchvision.transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from mpid_data import mpid_data_binary
from mpid_net import mpid_net_binary, mpid_func_binary

train_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(torch.cuda.is_available())

from lib.config import config_loader
#CFG = os.path.join("../../cfg","inference_config.cfg")
CFG = "/hepgpu4-data1/yuliia/MPID_repo/DM-CNN/cfg/inference_config.cfg"
cfg  = config_loader(CFG)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=cfg.GPUID

input_csv = pd.read_csv("/hepgpu4-data1/yuliia/training_output/MPID_scores_mpid_cosmics_mc_test_9291_steps.csv")

entry_of_interest = 2194

weight_file="/hepgpu4-data1/yuliia/MPID_pytorch/weights/mpid_model_20220716-04_20_PM_epoch_5_batch_id_1541_labels_2_title_0.001_AG_GN_final_step_9291.pwf"

mpid = mpid_net_binary.MPID()
mpid.cuda() 
mpid.load_state_dict(torch.load(weight_file, map_location=train_device))
mpid.eval()

test_file = "/hepgpu4-data1/yuliia/MPID/larcv2/mpid_cosmics_mc_test.root"
test_data = mpid_data_binary.MPID_Dataset(test_file, "image2d_image2d_binary_tree", train_device, plane=0, augment=False)

test_size = len(test_data)
print(len(input_csv))

print("\ntest_size = ", test_size)

test_loader = DataLoader(dataset=test_data, batch_size=cfg.batch_size_test, shuffle=True)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Scanning Occlusion ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

occlusion_step = 4

do_occlusion = True
entry_start = entry_of_interest
entries = 1
if not do_occlusion:
    entries = 10

run = []
subrun = []
event = []
energy = []
correct_prediction = []

for ENTRY in xrange(entry_start, entry_start + entries):
    print("ENTRY: ", ENTRY)

    input_image = test_data[ENTRY][0].view(-1,1,512,512)
    input_image[0][0][input_image[0][0] > 500] = 500
    input_image[0][0][input_image[0][0] < 10 ] = 0
    
    run.append(test_data[ENTRY][2][0])
    subrun.append(test_data[ENTRY][2][1])
    event.append(test_data[ENTRY][2][2])

    score = nn.Sigmoid()(mpid(input_image.cuda()))
    print(score[0][0])
    fig, ax= plt.subplots(1,1,figsize=(7,6))
    ax.imshow(input_image.cpu()[0][0], cmap='jet')
    ax.set_xlim(100,500)
    ax.set_ylim(100,500)
    ax.tick_params(top=0, bottom=0, left=0, right=0, labelleft=0, labelbottom=0)
    plt.savefig('/hepgpu4-data1/yuliia/training_output/this_image'+str(ENTRY)+'.png')
        
    if not do_occlusion: continue
    
    score_map_signal = np.full([512-occlusion_step, 512-occlusion_step], score.cpu().detach().numpy()[0][0])
    score_map_bkg = np.full([512-occlusion_step, 512-occlusion_step], score.cpu().detach().numpy()[0][1])
    
    clone_image = input_image.cpu().clone()
    for x in xrange(512 - occlusion_step):
        for y in xrange(512 - occlusion_step):
            clone_image = input_image.cpu().clone()

            if clone_image[0][0][x,y]==0:continue
            
            #print(np.shape(clone_image[0][0])) # (512,512)
            #print(np.shape(clone_image[0][0][x-occlusion_step:x+occlusion_step+1, y-occlusion_step:y+occlusion_step+1])) # (0,9)
            #print(np.shape(torch.zeros([2*occlusion_step+1, 2*occlusion_step+1]))) # (9,9)

            #print(x-occlusion_step)
            #print(x+occlusion_step+1)

            #if (x-occlusion_step < 0):
            #    x_new = 0
            #else:
            #    x_new = x-occlusion_step
            #x_start = x-occlusion_step
            #f = lambda x_start: (abs(x_start) + x_start) / 2

            clone_image[0][0][x-occlusion_step:x+occlusion_step+1,
                              y-occlusion_step:y+occlusion_step+1] = torch.zeros([2*occlusion_step+1, 2*occlusion_step+1])
            
            score = nn.Sigmoid()(mpid(clone_image.cuda())).cpu().detach().numpy()[0]
            score_map_signal[x,y] = score[0]
            score_map_bkg[x,y] = score[1]
        
def score_plot(score_map, title, xmin, xmax, ymin, ymax, vmin, vmax, cmap="jet"):

    if not xmin:
        xmin=0
        ymin=0
        xmax=score_map.shape[1]-1
        ymax=score_map.shape[1]-1
    
    xmin = 100
    xmax = 500
    ymin = 100
    ymax = 500
    
    fig, ax = plt.subplots(1, 1, figsize=(8,6))
    pos = ax.imshow(score_map, cmap=cmap, vmin=vmin, vmax=vmax)    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("%s score map"%title, fontsize=15)
    fig.colorbar(pos, ax=ax)
    ax.tick_params(top=0, bottom=0, left=0, right=0, labelleft=0, labelbottom=0)

    plt.savefig("/hepgpu4-data1/yuliia/training_output/%s_score_map.png"%title, bbox_inches="tight", pad_inches=0.01)

cmap="Oranges"
cmap="jet"

score_plot(score_map_signal, "signal", 0,0,0,0, np.min(score_map_signal),np.max(score_map_signal), cmap)
score_plot(score_map_bkg, "bkg", 0,0,0,0, np.min(score_map_bkg),np.max(score_map_bkg), cmap)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Box Occlusion ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

do_occlusion = True
entry_start = entry_of_interest
entries = 1
if not do_occlusion:
    entries = 10
i = 0
for ENTRY in xrange(entry_start, entry_start + entries):

    input_image = test_data[ENTRY][0].view(-1,1,512,512)

    score = nn.Sigmoid()(mpid(input_image.cuda()))
    
    fig, ax = plt.subplots(1, 1, figsize=(10,8))
    ax.imshow(input_image.cpu()[0][0], cmap='jet')
    
    ybase = 250
    xbase = 300
    ax.text(xbase+0,ybase+20, "signal, %.3f"%score.cpu().detach().numpy()[0][0],color="white",fontsize=15)
    ax.text(xbase+0,ybase+40, "bkg, %.3f"%score.cpu().detach().numpy()[0][1],color="white",fontsize=15)

    ax.text(xbase+0,ybase+120, "Entry, %i"%ENTRY,color="white",fontsize=15)

    run.append(int(test_data[ENTRY][2][0])) # int() to avoid long type
    subrun.append(int(test_data[ENTRY][2][1]))
    event.append(int(test_data[ENTRY][2][2]))

    ax.text(xbase+0,ybase+160, "True, %s"%run[-1],color="white",fontsize=15)

    plt.savefig("/hepgpu4-data1/yuliia/training_output/box_occlusion"+str(ENTRY)+".png", bbox_inches="tight", pad_inches=0.01)
    i += 1
"""entry_start=0
entries=1000
scores = np.zeros([5,1000])

for ENTRY in xrange(entry_start, entry_start + entries):
    input_image = test_data[ENTRY][0].view(-1,1,512,512)

    input_image[0][0][input_image[0][0] > 500] = 500
    input_image[0][0][input_image[0][0] < 10 ] = 0
    
    score = nn.Sigmoid()(mpid(input_image.cuda()))
    
    scores[0][ENTRY] = score.cpu().detach().numpy()[0][0]
    scores[1][ENTRY] = score.cpu().detach().numpy()[0][1]
    scores[2][ENTRY] = score.cpu().detach().numpy()[0][2]

fig,ax=plt.subplots(1,1,figsize=(8,6))
bins=np.linspace(0,1,20)
ax.hist(scores[0], bins=bins, label="elec", stacked=1)
ax.hist(scores[1], bins=bins, label="gamm", stacked=1)
ax.hist(scores[2], bins=bins, label="muon", stacked=1)

ax.legend()
plt.savefig("/hepgpu4-data1/yuliia/training_output/scores_hist_occlusion.png", bbox_inches="tight", pad_inches=0.01)"""