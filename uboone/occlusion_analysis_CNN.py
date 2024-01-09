import os, sys, ROOT
import uproot as np                                                    
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

# MPID scripts 
from mpid_data import mpid_data_binary
from mpid_net import mpid_net_binary, mpid_func
from lib.config import config_loader


plt.ioff()
torch.cuda.is_available()



def add_mask(input_tensor):
    input_tensor[input_tensor>0] = 0 
    return input_tensor



def score_plot(score_map, tag, title,vmin, vmax, cmap_input="gnuplot_r"):
    output_dir = "/hepgpu6-data1/lmlepin/outputs/"
    
    fig, ax = plt.subplots(1,1,figsize=(20,20),dpi=200)
    pos = ax.imshow(score_map.transpose(),origin="lower", cmap=cmap_input, vmin=vmin, vmax=vmax)     
    ax.set_xlabel("%s Score Map"%title, fontsize=35,labelpad=20)
    cbar = fig.colorbar(pos, ax=ax)
    cbar.ax.tick_params(labelsize=20)
    ax.tick_params(top=0, bottom=0, left=0, right=0, labelleft=0, labelbottom=0)

    plt.savefig(output_dir + "occlusion_test_{}_{}_map.png".format(tag,title),bbox_inches="tight")
    plt.savefig(output_dir + "occlusion_test_{}_{}_map.pdf".format(tag,title),bbox_inches="tight")


def RunOcclusion(input_file,input_entry,occlusion_size=4,normalized=True):
    MPID_PATH = os.path.dirname(mpid_data_binary.__file__)+"/../cfg"
    CFG = os.path.join(MPID_PATH,"inference_config_binary.cfg")
    cfg  = config_loader(CFG)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=cfg.GPUID
    weight_file="/hepgpu6-data1/lmlepin/CNN_weights/binary_cosmics_FULL_weights/mpid_model_COSMICS_FULL_20210819-08_34_PM_epoch_4_batch_id_581_labels_2_title_0.001_AG_GN_final_2_classes_step_8441.pwf"
    train_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mpid = mpid_net_binary.MPID()
    mpid.cuda()
    mpid.load_state_dict(torch.load(weight_file, map_location=train_device))
    mpid.eval()
    test_data = mpid_data_binary.MPID_Dataset(input_file,"image2d_image2d_binary_tree", train_device)
    test_loader = DataLoader(dataset=test_data, batch_size= 1 , shuffle=True)

    ### Scanning occlusion 
    occlusion_step = occlusion_size
    entry_start=input_entry 
    entries=1

    for ENTRY in xrange(entry_start, entry_start + entries):
        input_image = test_data[ENTRY][0].view(-1,1,512,512)
        input_image[0][0][input_image[0][0] > 500] = 500
        input_image[0][0][input_image[0][0] < 10 ] = 0
        
        score = nn.Sigmoid()(mpid(input_image.cuda()))
    

        output_dir = "/hepgpu6-data1/lmlepin/outputs/"
        fig, ax= plt.subplots(1,1,figsize=(22,22),dpi=200)
        ax.imshow(input_image.cpu()[0][0], cmap='jet')
        ax.set_xlim(100,500)
        ax.set_ylim(100,500)
        ax.tick_params(top=0, bottom=0, left=0, right=0, labelleft=0, labelbottom=0)
        ax.text(50,50, "Signal score: %.3f"%score.cpu().detach().numpy()[0][0],color="white",fontsize=53)
        plt.savefig(output_dir + "test_occlusion_PLANE.png")


        
        score_map_signal = np.full([512-occlusion_step, 512-occlusion_step], score.cpu().detach().numpy()[0][0])
        score_map_background = np.full([512-occlusion_step, 512-occlusion_step], score.cpu().detach().numpy()[0][1])

        clone_image = input_image.cpu().clone()
        
        for x in xrange(0 + occlusion_step, 512 - occlusion_step):
            for y in xrange(0 + occlusion_step,512 - occlusion_step):
                clone_image = input_image.cpu().clone()

                if(clone_image[0][0][x,y]==0):
                    continue
                

                clone_image[0][0][x-occlusion_step:x+occlusion_step+1,
                                y-occlusion_step:y+occlusion_step+1] = torch.zeros([2*occlusion_step+1, 2*occlusion_step+1])
                
                score = nn.Sigmoid()(mpid(clone_image.cuda())).cpu().detach().numpy()[0]
                score_map_signal[x,y] = score[0]
                score_map_background[x,y] = score[1]
    

    # Make plots 

    if(normalized):
        vmin_signal = np.min(score_map_signal)
        vmax_signal = np.max(score_map_signal)

        delta = vmax_signal - vmin_signal
        score_map_signal_norm = (score_map_signal - vmin_signal)/(delta)


        vmin_background = np.min(score_map_background)
        vmax_background = np.max(score_map_background)
        delta_background = vmax_background - vmin_background
        score_map_background_norm = (score_map_background - vmin_background)/(delta_background)

        vmin_signal_final = 0.
        vmax_signal_final = 1.0

        vmin_background_final = 0.
        vmax_background_final = 1.0
    else:
        score_map_signal_norm = score_map_signal 
        vmin_signal_final = np.min(score_map_signal)
        vmax_signal_final  = np.max(score_map_signal)
        
        score_map_background_norm = score_map_background
        vmin_background_final = np.min(score_map_background)
        vmax_background_final  = np.max(score_map_background)
        

    image_tag = "run1_NuMI_nu_overlay_larcv_cropped.root_{}".format(input_entry)
    score_plot(score_map_signal_norm,image_tag,"Signal",vmin_signal_final,vmax_signal_final,cmap_input='gnuplot_r')
    score_plot(score_map_background_norm, image_tag, "Background",vmin_background_final,vmax_background_final)




# Run occlusion 
base_dir = "/hepgpu6-data1/lmlepin/datasets/larcv/run1_samples/"
RunOcclusion(base_dir + "run1_NuMI_nu_overlay_larcv_cropped.root",200,occlusion_size=40,normalized=True)