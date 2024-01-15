# To run this script use the prod_dl_larcv2.sif container 

import os, sys, ROOT                                                   
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cbook as cbook 

plt.ioff()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataloader as dataloader
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from mpid_data import mpid_data_binary
from mpid_net import mpid_net_binary, mpid_func
from lib.config import config_loader
from scipy.ndimage import rotate
from scipy.ndimage import shift
import pandas as pd

def read_entry_list(filename):
    '''
    Function to read a set of entries 
    from a csv file. 

    filename: path to csv file

    returns an list containing the 
    entries stored in the csv file
    
    '''
    df = pd.read_csv(filename)
    entry_list = df['entry_number'].values.tolist()
    return entry_list



def logit_transform(score):
    return np.log(score/(1-score))

def PrintImage(input_file,ENTRY,run,signal=False,logit=False,data=False,input_angle=0,colorbar_included=False,logo=False):
    output_dir = "/hepgpu6-data1/lmlepin/outputs/"
    MPID_PATH = os.path.dirname(mpid_data_binary.__file__)+"/../cfg"
    CFG = os.path.join(MPID_PATH,"inference_config_binary.cfg")
    cfg  = config_loader(CFG)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=cfg.GPUID

    weight_file="/hepgpu6-data1/lmlepin/CNN_weights/binary_cosmics_FULL_weights/mpid_model_COSMICS_FULL_20210819-08_34_PM_epoch_4_batch_id_581_labels_2_title_0.001_AG_GN_final_2_classes_step_8441.pwf"
    train_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mpid = mpid_net_binary.MPID()
    mpid.load_state_dict(torch.load(weight_file, map_location=train_device))
    mpid.eval()


    if(signal):
        test_file = "/hepgpu6-data1/lmlepin/datasets/" + input_file
        #test_file = "/hepgpu6-data1/lmlepin/datasets/larcv/{}_signal/".format(run) + input_file 
    else:
        test_file = "/hepgpu6-data1/lmlepin/datasets/larcv/" + input_file
        #test_file = "/hepgpu6-data1/lmlepin/datasets/larcv/{}_samples/".format(run) + input_file 



    test_data = mpid_data_binary.MPID_Dataset(test_file,"image2d_image2d_binary_tree", train_device)
    test_loader = DataLoader(dataset=test_data, batch_size= 1 , shuffle=True)
    input_image = test_data[ENTRY][0].view(-1,1,512,512)
    event_info = test_data[ENTRY][2]
    input_image[0][0][input_image[0][0] > 500] = 500
    input_image[0][0][input_image[0][0] < 10 ] = 0
    #input_image[0][0] = torch.tensor(shift(input_image[0][0],[-100,100]))

    if(input_angle > 0.):
        input_image[0][0] = torch.tensor(rotate(input_image[0][0],angle=input_angle))
    score = nn.Sigmoid()(mpid(input_image))
    fig, ax= plt.subplots(1,1,figsize=(20,20),dpi=300)
    #plt.title("MicroBooNE Simulation",fontsize=53,pad=20)
    img = ax.imshow(input_image.cpu()[0][0].detach().cpu().numpy().transpose(),origin="lower",cmap='jet',norm=colors.PowerNorm(gamma=0.35,vmin=input_image.cpu()[0][0].min(), vmax=input_image.cpu()[0][0].max()))
    



    if(colorbar_included):
        newax_cb = fig.add_axes([0.0001,0.3,0.2,0.4], anchor='NE', zorder=1)
        cbar = fig.colorbar(img, ax=newax_cb,shrink=0.7,cmap='jet')
        cbar.set_ticks([0.,np.max(input_image.cpu()[0][0].detach().cpu().numpy())])
        cbar.ax.set_yticklabels(['Low charge', 'High charge'],color='white',fontsize=50) 
        newax_cb.axis('off')

    if(logo):
        plt.rcParams["figure.figsize"] = [7.00, 3.50]
        #plt.rcParams["figure.autolayout"] = True
        im = plt.imread('/hepgpu6-data1/lmlepin/datasets/kalekologo_noshadow_notrack_inverted.png') # insert local path of the image.
        ax.plot(range(10))
        newax = fig.add_axes([0.15,0.58,0.3,0.3], anchor='NE', zorder=1)
        newax.imshow(im)
        newax.axis('off')

    ax.set_xticks([0, 511])
    ax.set_yticks([511])

    ax.tick_params(axis="y", which='major', direction="out",length=10,width=2.5,pad=10, labelsize=50)
    ax.tick_params(axis="y", which='minor', direction="out",length=10,width=1.0,labelleft=False, labelsize=50)
    ax.tick_params(axis="x", which='major', direction="out",length=10,width=2.5,pad=10, bottom=True,top=False, labelsize=50)
    ax.tick_params(axis="x", which='minor', direction="out",length=10,width=2.0, bottom=True,top=False, labelsize=50)

    ax.set_xlim(0,511)
    ax.set_ylim(0,511)

    ax.set_xlabel('Wire Number', size=55,labelpad=1.0)
    ax.set_ylabel('Drift Time', size=55,labelpad=1.0)


    if(data):
        ax.text(20,80, "MicroBooNE NuMI Data", color="white",fontsize=53)
        ax.text(20,20,"Run: {}, Subrun: {}, Event {}".format(event_info[0],event_info[1],event_info[2]),color="white",fontsize=53)
    
    if(signal):
        ax.text(20,20, "Dark Trident Simulation", color="white",fontsize=53)

    elif(not signal and not data):
        ax.text(20,20, "Neutrino Background Simulation", color="white",fontsize=53)


    if(logit):
        ax.text(20,50, "Signal score: %.3f"%logit_transform(score.cpu().detach().numpy()[0][0]),color="white",fontsize=53)
    else:
        pass
        #ax.text(20,80, "Energy: %.3f GeV"%0.880, color="white",fontsize=53)
        #ax.text(20,50, "Signal score: %.3f"%score.cpu().detach().numpy()[0][0],color="white",fontsize=53)
    #ax.axis('off')


    output_name_base = os.path.splitext(input_file)[0] + "_ENTRY_" + str(ENTRY) 

    if(input_angle > 0.):
        output_name_base += "_rotated_" + str(input_angle) 


    if(colorbar_included):
        ouput_name_pdf = output_name_base +  "_colorbar_logit.png"
        ouput_name_png = output_name_base +  "_colorbar_logit.pdf"
    else:
        ouput_name_pdf = output_name_base +  ".png"
        ouput_name_png = output_name_base +  ".pdf"


    plt.savefig(output_dir + ouput_name_png,bbox_inches="tight")
    plt.savefig(output_dir + ouput_name_pdf,bbox_inches="tight")




decay_modes = ["pi0","eta"]
mass_ratio=["0.6"]
masses_pi0 = ["0.01", "0.05"]
#masses_pi0 = ["0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08", "0.09"]
#masses_eta = ["0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08", "0.09", "0.1","0.2","0.3","0.4"]


PrintImage("NCPi0_production_v2_cropped.root",1400,"run1",signal=False,data=False,input_angle=0,colorbar_included=True,logo=True)
PrintImage("NCPi0_production_v2_cropped.root",1500,"run1",signal=False,data=False,input_angle=0,colorbar_included=True,logo=True)
PrintImage("NCPi0_production_v2_cropped.root",1600,"run1",signal=False,data=False,input_angle=0,colorbar_included=True,logo=True)
PrintImage("NCPi0_production_v2_cropped.root",1700,"run1",signal=False,data=False,input_angle=0,colorbar_included=True,logo=True)
PrintImage("NCPi0_production_v2_cropped.root",1800,"run1",signal=False,data=False,input_angle=0,colorbar_included=True,logo=True)

'''

csv_dir = "/hepgpu6-data1/lmlepin/datasets/evd_list/"

for m in masses_pi0:
    temp_file  = csv_dir + "cnn_bdt_dt"+mass_ratio[0] + "_ma"+m+"_pi0.csv"
    temp_list = read_entry_list(temp_file) 
    print(temp_list)
    temp_root = "run1_dt_ratio_{}_ma_{}_pi0_larcv_cropped.root".format(mass_ratio[0],m)
    for entry in temp_list:
        PrintImage(temp_root,int(entry),"run1",signal=True,data=False,input_angle=0,colorbar_included=False,logo=True)



for m in masses_pi0:
    temp_file  = csv_dir + "bdt_cnn_dt"+mass_ratio[0] + "_ma"+m+"_pi0.csv"
    temp_list = read_entry_list(temp_file) 
    print(temp_list)
    temp_root = "run1_dt_ratio_{}_ma_{}_pi0_larcv_cropped.root".format(mass_ratio[0],m)
    for entry in temp_list:
        PrintImage(temp_root,int(entry),"run1",signal=True,data=False,input_angle=0,colorbar_included=False,logo=True)


for m in masses_eta:
    temp_file  = csv_dir + "cnn_bdt_dt"+mass_ratio[0] + "_ma"+m+"_eta.csv"
    temp_list = read_entry_list(temp_file) 
    for entry in temp_list:
        temp_root = "run1_dt_ratio_{}_ma_{}_eta_larcv_cropped.root".format(mass_ratio[0],m)
        PrintImage(temp_root,entry,"run1",signal=True,data=False,input_angle=0,colorbar_included=False,logo=True)



run3_entries = [1200, 1337, 2162, 3138, 3161, 3653, 5135, 6902]
run1_entries = [363, 737, 1049, 1278, 2889, 4204]

run_entries = [10, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]

run_entries_bkg = [200, 377, 500, 800, 1000, 1100, 1200, 1300]


PrintImage("run1_dt_ratio_0.6_ma_0.05_pi0_larcv_cropped.root",1200,"run1",signal=True,data=False,input_angle=90,colorbar_included=False,logo=True)
PrintImage("run1_dt_ratio_0.6_ma_0.05_pi0_larcv_cropped.root",1200,"run1",signal=True,data=False,input_angle=180,colorbar_included=False,logo=True)
PrintImage("run1_dt_ratio_0.6_ma_0.05_pi0_larcv_cropped.root",1200,"run1",signal=True,data=False,input_angle=270,colorbar_included=False,logo=True)


for entry in run_entries_bkg:
    PrintImage("run1_NuMI_nu_overlay_larcv_cropped.root", entry, "run1",signal=False,data=False,input_angle=0,colorbar_included=False,logo=True)



for entry in run_entries:
    PrintImage("run1_dt_ratio_0.6_ma_0.3_eta_larcv_cropped.root",entry,"run1",signal=True,data=False,input_angle=0,colorbar_included=False,logo=True)


for entry in run1_entries: 
    PrintImage("run1_NuMI_beamon_larcv_cropped.root",entry,"run1",signal=False,data=True,logit=True,input_angle=0,colorbar_included=True,logo=True)

for entry in run3_entries: 
    PrintImage("run3_NuMI_beamon_larcv_cropped.root",entry,"run3",signal=False,data=True,logit=True,input_angle=0,colorbar_included=True,logo=True)

''' 