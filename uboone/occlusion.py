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

train_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(torch.cuda.is_available())

from lib.config import config_loader
#CFG = os.path.join("../../cfg","inference_config.cfg")
CFG = "/hepgpu4-data1/yuliia/MPID_repo/DM-CNN/cfg/inference_config.cfg"
cfg  = config_loader(CFG)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=cfg.GPUID

input_csv = pd.read_csv("/hepgpu4-data1/yuliia/training_output/test_additional_5099_steps.csv")

entry_of_interest = 54

weight_file="/hepgpu4-data1/yuliia/MPID_pytorch/weights/mpid_model_20220706-06_45_PM_epoch_3_batch_id_1061_title_0.001_AG_GN_final_step_5099.pwf"

mpid = mpid_net.MPID()
mpid.cuda() 
mpid.load_state_dict(torch.load(weight_file, map_location=train_device))
mpid.eval()

test_file = "/hepgpu4-data1/yuliia/MPID/larcv2/test_set_collection_plane_2.root"
test_data = mpid_data.MPID_Dataset(test_file, "image2d_image2d_tree", train_device, nclasses=cfg.num_class, plane=0, augment=cfg.augment)

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
    
    run.append(test_data[ENTRY][2])
    subrun.append(test_data[ENTRY][3])
    event.append(test_data[ENTRY][4])

    score = nn.Sigmoid()(mpid(input_image.cuda()))

    # Is the prediction correct? No -- keep, Yes -- move on
    #print(score)
    #print(subrun[-1])

    if (subrun[-1] == 11): 
        true = torch.tensor([1.,0.,0.])
    elif (subrun[-1] == 22): 
        true = torch.tensor([0.,1.,0.])
    else:
        true = torch.tensor([0.,0.,1.])

    if (torch.argmax(score).item() == 0):
        predicted = torch.tensor([1.,0.,0.])
    elif (torch.argmax(score).item() == 1): 
        predicted = torch.tensor([0.,1.,0.])
    else:
        predicted = torch.tensor([0.,0.,1.])

    if (torch.eq(predicted,true)[0].item() == True & torch.eq(predicted,true)[1].item() == True & torch.eq(predicted,true)[2].item() == True):
        correct_prediction.append(True)
        print('Correct prediction')
        continue
    else:
        correct_prediction.append(False)
        print('Incorrect prediction')

        fig, ax= plt.subplots(1,1,figsize=(7,6))
        ax.imshow(input_image.cpu()[0][0], cmap='jet')
        ax.set_xlim(100,500)
        ax.set_ylim(100,500)
        ax.tick_params(top=0, bottom=0, left=0, right=0, labelleft=0, labelbottom=0)
        plt.savefig('/hepgpu4-data1/yuliia/training_output/this_image'+str(ENTRY)+'.png')
    #     ax.text(0,20, "elec, %.3f"%score.cpu().detach().numpy()[0][0],color="white",fontsize=15)
    #     ax.text(0,40, "gamm, %.3f"%score.cpu().detach().numpy()[0][1],color="white",fontsize=15)
    #     ax.text(0,60, "muon, %.3f"%score.cpu().detach().numpy()[0][2],color="white",fontsize=15)
    #     ax.text(0,120, "Entry, %i"%ENTRY,color="white",fontsize=15)
    #     ax.text(0,140, "eng elec, %.1f"%(engs[0]*1000),color="white",fontsize=15)
    #     ax.text(0,160, "eng gam, %.1f"%(engs[1]*1000),color="white",fontsize=15)
    #     ax.text(0,180, "eng muo, %.1f"%(engs[2]*1000),color="white",fontsize=15)
        
        if not do_occlusion: continue
        
    #     clone_image = input_image.cpu().clone()
    #     clone_image[0][0][200:300, 400:430] = 50
    #     fig, ax= plt.subplots(1,1,figsize=(10,8))    
    #     ax.imshow(clone_image.cpu()[0][0], cmap='jet')
        score_map_elec = np.full([512-occlusion_step, 512-occlusion_step], score.cpu().detach().numpy()[0][0])
        score_map_gamm = np.full([512-occlusion_step, 512-occlusion_step], score.cpu().detach().numpy()[0][1])
        score_map_muon = np.full([512-occlusion_step, 512-occlusion_step], score.cpu().detach().numpy()[0][2])
        
        clone_image = input_image.cpu().clone()
        for x in xrange(512 - occlusion_step):
            for y in xrange(512 - occlusion_step):
                clone_image = input_image.cpu().clone()
    #             if np.count_nonzero(clone_image[0][0][x-occlusion_step:x+occlusion_step+1,
    #                                                   y-occlusion_step:y+occlusion_step+1])==0:continue
                if clone_image[0][0][x,y]==0:continue
                
                clone_image[0][0][x-occlusion_step:x+occlusion_step+1,
                                y-occlusion_step:y+occlusion_step+1] = torch.zeros([2*occlusion_step+1, 2*occlusion_step+1])
                
                score = nn.Sigmoid()(mpid(clone_image.cuda())).cpu().detach().numpy()[0]
                score_map_elec[x,y] = score[0]
                score_map_gamm[x,y] = score[1]
                score_map_muon[x,y] = score[2]
            
    #             fig, ax= plt.subplots(1,1,figsize=(10,8))    
    #             ax.imshow(clone_image[0][0], cmap='jet')
    #             ax.plot(y, x, "*", color = 'white',markersize=5)
    #             ax.text(0,20, score[0],color="white",fontsize=15)
    #             ax.text(0,40, score[1],color="white",fontsize=15)
    #             ax.text(0,60, score[2],color="white",fontsize=15)

    #             ax.text(0,0,"Y Plane", fontsize=20, color="white")
    #             ax.text(0, 50, "MicroBooNE Simulation", fontsize=20, color="white")

    #     clone_image = input_image.cpu().clone()
    # #     clone_image = torch.ones([1,1,512,512])
    #     y = 190
    #     x = 430
    #     clone_image[0][0][x-occlusion_step:x+occlusion_step+1,
    #                       y-occlusion_step:y+occlusion_step+1] = torch.zeros([2*occlusion_step+1, 2*occlusion_step+1])
    #     score = nn.Sigmoid()(mpid(clone_image.cuda())).cpu().detach().numpy()[0]
    #     fig, ax= plt.subplots(1,1,figsize=(10,8))    
    #     ax.imshow(clone_image[0][0], cmap='jet')
    #     ax.plot(y, x, "*", color = 'white',markersize=5)
    #     ax.text(0,20, score[0],color="white",fontsize=15)
    #     ax.text(0,40, score[1],color="white",fontsize=15)
    #     ax.text(0,60, score[2],color="white",fontsize=15)

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

score_plot(score_map_elec, "electron", 0,0,0,0, np.min(score_map_elec),np.max(score_map_elec), cmap)
score_plot(score_map_gamm, "photon", 0,0,0,0, np.min(score_map_gamm),np.max(score_map_gamm), cmap)
score_plot(score_map_muon, "muon", 0,0,0,0, np.min(score_map_muon),np.max(score_map_muon), cmap)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Box Occlusion ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""do_occlusion = True
entry_start = entry_of_interest
entries = 100
if not do_occlusion:
    entries = 10
i = 0
for ENTRY in xrange(entry_start, entry_start + entries):

    if (correct_prediction[i] == True):
        i += 1
        continue
    else:
        input_image = test_data[ENTRY][0].view(-1,1,512,512)

        score = nn.Sigmoid()(mpid(input_image.cuda()))
        
        fig, ax = plt.subplots(1, 1, figsize=(10,8))
        ax.imshow(input_image.cpu()[0][0], cmap='jet')
        
        ybase = 250
        xbase = 300
        ax.text(xbase+0,ybase+20, "elec, %.3f"%score.cpu().detach().numpy()[0][0],color="white",fontsize=15)
        ax.text(xbase+0,ybase+40, "gamm, %.3f"%score.cpu().detach().numpy()[0][1],color="white",fontsize=15)
        ax.text(xbase+0,ybase+60, "muon, %.3f"%score.cpu().detach().numpy()[0][2],color="white",fontsize=15)
        ax.text(xbase+0,ybase+120, "Entry, %i"%ENTRY,color="white",fontsize=15)
        ax.text(xbase+0,ybase+160, "True, %s"%subrun[i],color="white",fontsize=15)

        run.append(int(test_data[ENTRY][2])) # int() to avoid long type
        subrun.append(int(test_data[ENTRY][3]))
        event.append(int(test_data[ENTRY][4]))

        # Search the full input data by run, subrun & event to find the corresponding energy
        eng = input_csv.loc[(input_csv['run'] == run[-1]) & (input_csv['subrun'] == subrun[-1]) & (input_csv['event'] == event[-1]), 'Energy'].values[0]
        ax.text(xbase+0,ybase+140, "eng [MeV], %.1f"%(eng*1000),color="white",fontsize=15)

        plt.savefig("/hepgpu4-data1/yuliia/training_output/box_occlusion"+str(ENTRY)+".png", bbox_inches="tight", pad_inches=0.01)
        i += 1
entry_start=0
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