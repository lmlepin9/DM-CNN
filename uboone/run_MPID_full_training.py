# Standard python libraries
import os, sys, ROOT
import uproot as np
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 

# Pytorch stuff
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataloader as dataloader
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

# For rotations
from scipy.ndimage import rotate

# MPID scripts 
from mpid_data import mpid_data_binary
from mpid_net import mpid_net_binary, mpid_func

plt.ioff()
torch.cuda.is_available()

from lib.config import config_loader




def InferenceCNN(input_file, input_csv, output_tag, steps="8441", rotation=False,rotation_angle=0.):
    print("Processing with steps: ", steps)
    MPID_PATH = os.path.dirname(mpid_data_binary.__file__)+"/../cfg"
    CFG = os.path.join(MPID_PATH,"inference_config_binary.cfg")
    cfg  = config_loader(CFG)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=cfg.GPUID


    output_dir = "/hepgpu6-data1/lmlepin/outputs/"

    if(rotate):
        output_file = output_dir + output_tag + "_CNN_scores_{}_steps_rotation_{}.csv".format(steps,str(rotation_angle))
    else:
        output_file = output_dir + output_tag + "_CNN_scores_{}_steps.csv".format(steps)



    if(steps=="8441"):
        weight_file="/hepgpu6-data1/lmlepin/CNN_weights/binary_cosmics_FULL_weights/mpid_model_COSMICS_FULL_20210819-08_34_PM_epoch_4_batch_id_581_labels_2_title_0.001_AG_GN_final_2_classes_step_8441.pwf"
    elif(steps=="9241"):
        weight_file="/hepgpu6-data1/lmlepin/CNN_weights/binary_cosmics_FULL_weights/mpid_model_COSMICS_FULL_20210819-08_52_PM_epoch_4_batch_id_1381_labels_2_title_0.001_AG_GN_final_2_classes_step_9241.pwf"
    elif(steps=="11526"):
        weight_file="/hepgpu6-data1/lmlepin/CNN_weights/binary_cosmics_FULL_weights/mpid_model_COSMICS_FULL_20210819-09_42_PM_epoch_5_batch_id_1701_labels_2_title_0.001_AG_GN_final_2_classes_step_11526.pwf"
    train_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    df = pd.read_csv(input_csv)
    df['signal_score']=np.ones(len(df))*-999999.9
    df['entry_number']=np.ones(len(df))*-1
    df['n_pixels']=np.ones(len(df))*-1
    mpid = mpid_net_binary.MPID()
    mpid.cuda()
    mpid.load_state_dict(torch.load(weight_file, map_location=train_device))
    mpid.eval()
    test_data = mpid_data_binary.MPID_Dataset(input_file,"image2d_image2d_binary_tree", train_device)
    n_events = test_data[0][3]


    print("Total number of events: ", n_events)
    print("Starting...")

    for ENTRY in range(n_events - 1):
        if(ENTRY%1000 == 0):
            print("ENTRY: ", ENTRY)
            
        run_info = test_data[ENTRY][2][0]
        subrun_info = test_data[ENTRY][2][1]
        event_info = test_data[ENTRY][2][2]
        index_array = df.query('run_number == {:2d} & subrun_number == {:2d} & event_number == {:2d} '.format(run_info,subrun_info,event_info)).index.values
        input_image = test_data[ENTRY][0].view(-1,1,512,512)
        input_image[0][0][input_image[0][0] > 500] = 500
        input_image[0][0][input_image[0][0] < 10 ] = 0
        if(rotate):
            input_image[0][0] = torch.tensor(rotate(input_image[0][0],angle=rotation_angle))
        score = nn.Sigmoid()(mpid(input_image.cuda())).cpu().detach().numpy()[0]
        if(len(index_array) ==0):
            continue
        df['signal_score'][index_array[0]]=score[0] 
        df['entry_number'][index_array[0]]=ENTRY
        df['n_pixels'][index_array[0]]=np.count_nonzero(input_image)


    dp=df[df['signal_score'] >= 0.]                       
    df.to_csv(output_file,index=False)
    plt.figure()
    plt.hist(dp['signal_score'], bins = 40, alpha=0.9, label=output_tag,histtype='bar')
    plt.xlabel("Signal score")
    plt.grid()
    plt.legend(loc='upper right')

    if(rotate):
        plt.savefig(output_dir + output_tag +"_CNN_signal_score_distribution_{}_steps_rotation_{}.png".format(steps,str(rotation_angle)))
    else:
        plt.savefig(output_dir + output_tag +"_CNN_signal_score_distribution_{}_steps.png".format(steps))








def ProcessVariations(run):

    det_vars=[
        'CV',
        'LYAttenuation',
        'LYDown',
        'LYRayleigh',
        'Recomb',
        'SCE',
        'WireModdEdX',
        'WireModThetaXZ',
        'WireModThetaYZ_withouts',
        'WireModThetaYZ_withs',
        'WireModX',
        'WireModYZ'
    ]

    larcv_base_dir = "/hepgpu6-data1/lmlepin/datasets/larcv/" + run +"_det_var/"
    csv_base_dir = "/hepgpu6-data1/lmlepin/datasets/csv_files/"

    print("Run selected: " + run)

    for var in det_vars:
        print("Processing: " + var)
        larcv_file_name = run + "_" + var + "_larcv_cropped.root" 
        csv_file_name = run + "_" + var + "_CNN.csv"
        output_tag = run + "_" + var 
        InferenceCNN(larcv_base_dir + larcv_file_name, csv_base_dir + csv_file_name, output_tag)
    

def ProcessSamples(run,steps):

    samples = [
        "nu_overlay",
        #"dirt",
        #"offbeam",
        #"beamon"
    ]

    larcv_base_dir = "/hepgpu6-data1/lmlepin/datasets/larcv/" + run +"_samples/"
    csv_base_dir = "/hepgpu6-data1/lmlepin/datasets/csv_files/"

    print("Run selected: " + run)

    for sample in samples:
        print("Processing: " + sample)
        larcv_file_name = run + "_NuMI_" + sample + "_larcv_cropped.root" 
        csv_file_name = run + "_" + sample + "_CNN.csv"
        output_tag = run + "_" + sample 
        InferenceCNN(larcv_base_dir + larcv_file_name, csv_base_dir + csv_file_name, output_tag,steps)


def ProcessCV(run):
    larcv_file_name = "/hepgpu6-data1/lmlepin/datasets/larcv/" + run + "_det_var/" + run + "_CV_high_stats_larcv_cropped.root"
    csv_file_name = "/hepgpu6-data1/lmlepin/datasets/csv_files/" + run + "_CV_high_stats_CNN.csv"
    output_tag = run + "_CV_high_stats"
    InferenceCNN(larcv_file_name, csv_file_name, output_tag)


def ProcessSignalCorsika():
    larcv_file_name = "/hepgpu6-data1/lmlepin/datasets/larcv/dt_0.05_corsika_training_set_larcv_cropped.root"
    csv_file_name = "/hepgpu6-data1/lmlepin/datasets/csv_files/dt_0.05_corsika_training_set.csv"
    output_tag = "dt_0.05_corsika_test"
    InferenceCNN(larcv_file_name, csv_file_name, output_tag)


def ProcessSignal(run,ratio,steps):
    print("Processing signal samples of: " + run)
    print("Ratio: ", ratio)

    if(ratio == "0.6"):
        masses_pi0 = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
        masses_eta = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4]
    else:
        masses_pi0 = [0.010, 0.020, 0.030]
        masses_eta = [0.010, 0.020, 0.030, 0.040, 0.050, 0.060, 0.065,
                      0.070, 0.075, 0.080, 0.085, 0.090, 0.095, 0.100,
                      0.105, 0.110, 0.115, 0.120, 0.125]


    for mass in masses_pi0:
        if(ratio=="0.6"):
            mass = "{:.2f}".format(mass)
        else: 
            mass = "{:.3f}".format(mass)
        decay="pi0"
        print("Processing mass point: ", mass, "... decay mode: ", decay)
        larcv_file_name = "/hepgpu6-data1/lmlepin/datasets/larcv/{}_signal/{}_dt_ratio_{}_ma_{}_{}_larcv_cropped.root".format(run,run,ratio,mass,decay)
        csv_file_name = "/hepgpu6-data1/lmlepin/datasets/csv_files/{}_dt_ratio_{}_ma_{}_{}_CNN.csv".format(run,ratio,mass,decay)
        output_tag = "{}_dt_ratio_{}_{}_{}".format(run,ratio,mass,decay)
        InferenceCNN(larcv_file_name, csv_file_name, output_tag,steps)


    for mass in masses_eta:
        if(ratio=="0.6"):
            mass = "{:.2f}".format(mass)
        else: 
            mass = "{:.3f}".format(mass)
        decay="eta"
        print("Processing mass point: ", mass, "... decay mode: ", decay)
        larcv_file_name = "/hepgpu6-data1/lmlepin/datasets/larcv/{}_signal/{}_dt_ratio_{}_ma_{}_{}_larcv_cropped.root".format(run,run,ratio,mass,decay)
        csv_file_name = "/hepgpu6-data1/lmlepin/datasets/csv_files/{}_dt_ratio_{}_ma_{}_{}_CNN.csv".format(run,ratio,mass,decay)
        output_tag = "{}_dt_ratio_{}_{}_{}".format(run,ratio,mass,decay)
        InferenceCNN(larcv_file_name, csv_file_name, output_tag,steps)
  


def ProcessSignalRotation(run,ratio,steps):
    print("Processing signal samples of: " + run)
    print("Ratio: ", ratio)

    masses_pi0 = [0.05]
    rotations = [90, 180, 270]

    for mass in masses_pi0:

        if(ratio=="0.6"):
            mass = "{:.2f}".format(mass)
        else: 
            mass = "{:.3f}".format(mass)
        decay="pi0"
        print("Processing mass point: ", mass, "... decay mode: ", decay)
        larcv_file_name = "/hepgpu6-data1/lmlepin/datasets/larcv/{}_signal/{}_dt_ratio_{}_ma_{}_{}_larcv_cropped.root".format(run,run,ratio,mass,decay)
        csv_file_name = "/hepgpu6-data1/lmlepin/datasets/csv_files/{}_dt_ratio_{}_ma_{}_{}_CNN.csv".format(run,ratio,mass,decay)
        output_tag = "{}_dt_ratio_{}_{}_{}".format(run,ratio,mass,decay)

        for rotation in rotations:
            print("Processing rotation: ", rotation)
            InferenceCNN(larcv_file_name, csv_file_name, output_tag,steps,rotation=True,rotation_angle=rotation)


def ProcessDirtOverlay():
    larcv_file_name = "/hepgpu6-data1/lmlepin/datasets/larcv/run1_NuMI_dirt_larcv_cropped.root"
    csv_file_name = "/hepgpu6-data1/lmlepin/datasets/csv_files/run1_dirt_CNN.csv"
    output_tag = "run1_NuMI_dirt"
    InferenceCNN(larcv_file_name, csv_file_name, output_tag)


def ProcessSignalVars(run, mass):

    det_vars = ["CV", "ly_atten", "ly_down", "ly_rayleigh", "recomb", "sce", "wiremod_anglexz", "wiremod_angleyz", "wiremod_dEdx", "wiremod_x", "wiremod_yz"]

    larcv_base_dir = "/hepgpu6-data1/lmlepin/datasets/larcv/" + run +"_signal_det_var/"
    csv_base_dir = "/hepgpu6-data1/lmlepin/datasets/csv_files/"

    print("Run selected: " + run)

    for var in det_vars:
        print("Processing: " + var)
        larcv_file_name = run + "_ratio_0.6_ma_" + str(mass) + "_" + var + "_larcv_cropped.root" 
        csv_file_name = run + "_dt_ratio_0.6_ma_" + str(mass) + "_" + var + "_CNN.csv"
        output_tag = run + "_dt_ratio_0.6_ma_" + str(mass) + "_" + var 
        InferenceCNN(larcv_base_dir + larcv_file_name, csv_base_dir + csv_file_name, output_tag)

if __name__ == "__main__":
    #ProcessSamples("run1","8441")
    ProcessSamples("run3","8441")
    #ProcessSignalRotation("run1","0.6","8441")
    #ProcessSignal("run1","0.6","8441")
    #ProcessSignal("run3","0.6","8441")
     #ProcessSignal("run1","0.05","pi0")
   # ProcessSignal("run1","0.05","eta")
   # ProcessSignal("run1","0.1","eta")
   # ProcessSignal("run1","0.2","eta")
   # ProcessSignal("run1","0.3","eta")
   # ProcessSignal("run1","0.4","eta")
   # ProcessVariations("run3")
   # ProcessSignalVars("run3",0.01)
   # ProcessSignalVars("run3",0.05)
   # ProcessSignalVars("run3",0.4)