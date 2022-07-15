
from __future__ import division
from __future__ import print_function


import torch
# We set this here, otherwise pytorch can't recognize CUDA
print("Checking if CUDA is availbale: ")
print(torch.cuda.is_available())
print("\n")



import os, sys
from lib.config import config_loader
from lib.utility import timestr

BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
CFG = os.path.join(BASE_PATH,"../cfg","simple_config.cfg")
cfg  = config_loader(CFG)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=cfg.GPUID

import numpy as np


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.dataloader as dataloader

from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from mpid_data import mpid_data_binary
from mpid_net import mpid_net_binary, mpid_func_binary

title = cfg.name
#if (len(sys.argv) > 1) : title = sys.argv[1]

SEED = 1
cuda = torch.cuda.is_available()

# # For reproducibility
# torch.manual_seed(SEED)

# if cuda:
#     torch.cuda.manual_seed(SEED)

print ("Using {} GPUs".format(torch.cuda.device_count()))


train_device = 'cuda' if torch.cuda.is_available() else 'cpu'
#test_device = 'cuda:1' if torch.cuda.is_available() else 'cpu'


# Training data
train_file = "/hepgpu5-data1/yuliia/MPID/larcv2/mpid_cosmics_mc_training.root"
#train_file = "/scratch/ruian/training_data/MPID/larcv2/train_normal/larcv_7e19963d-1018-42b5-940c-48489454fa1e.root"
train_data = mpid_data_binary.MPID_Dataset(train_file, "image2d_image2d_binary_tree", train_device, plane=0, augment=False)
train_loader = DataLoader(dataset=train_data, batch_size=cfg.batch_size_train, shuffle=True)
labels = 2

# Test data
test_file = "/hepgpu5-data1/yuliia/MPID/larcv2/mpid_cosmics_mc_test.root"
test_data = mpid_data_binary.MPID_Dataset(test_file, "image2d_image2d_binary_tree", train_device, plane=0)
test_loader = DataLoader(dataset=test_data, batch_size=cfg.batch_size_test, shuffle=True)

mpid = mpid_net_binary.MPID(dropout=cfg.drop_out, num_classes=2)
mpid.cuda()

# Using BCEWithLogitsLoss instead of 
# Using Sigmoid in mpidnet + BCELoss 
#loss_fn = nn.BCELoss()
loss_fn = nn.BCEWithLogitsLoss()


optimizer  = optim.Adam(mpid.parameters(), lr=cfg.learning_rate)#, weight_decay=0.001)
train_step = mpid_func_binary.make_train_step(mpid, loss_fn, optimizer)
test_step  = mpid_func_binary.make_test_step(mpid, test_loader, loss_fn, optimizer)

print ("Training with {} images".format(len(train_loader.dataset)))

train_losses = []
train_accuracies =[]
test_losses = []
test_accuracies =[]

EPOCHS = cfg.EPOCHS

fout = open('/hepgpu5-data1/yuliia/training_csvs/production_{}_{}.csv'.format(timestr(), title), 'w')
fout.write('train_accu,test_accu,train_loss,test_loss,epoch,step')
fout.write('\n')

print ("Start training process................")

step=0

for epoch in range(EPOCHS):
    print ("\n")
    print (" @{}th epoch...".format(epoch))
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        print ("\n")
        # the dataset "lives" in the CPU, so do our mini-batches
        # therefore, we need to send those mini-batches to the
        # device where the model "lives"
        print (" @{}th epoch, @ batch_id {}".format(epoch, batch_idx))
        
        x_batch = x_batch.to(train_device).view((-1,1,512,512))
        y_batch = y_batch.to(train_device)
                
        loss = train_step(x_batch, y_batch) #model.train() called in train_step
        train_losses.append(loss)
            
        print('\r Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch,
            EPOCHS-1,
            batch_idx * len(x_batch), 
            len(train_loader.dataset),
            100. * batch_idx / len(train_loader), 
            loss), 
            end='')


        if (batch_idx % cfg.test_every_step == 1 and cfg.run_test):
            if (cfg.save_weights and epoch >= 3 and epoch <= 6):
                torch.save(mpid.state_dict(), "/hepgpu5-data1/yuliia/MPID_pytorch/weights/mpid_model_{}_epoch_{}_batch_id_{}_labels_{}_title_{}_step_{}.pwf".format(timestr(), epoch, batch_idx,labels, title, step))

            print ("Start eval on test sample.......@step..{}..@epoch..{}..@batch..{}".format(step,epoch, batch_idx))
            test_accuracy = mpid_func_binary.validation(mpid, test_loader, cfg.batch_size_test, train_device, event_nums=cfg.test_events_nums)
            print ("Test Accuracy {}".format(test_accuracy))
            print ("Start eval on training sample...@epoch..{}.@batch..{}".format(epoch, batch_idx))
            train_accuracy = mpid_func_binary.validation(mpid, train_loader, cfg.batch_size_train, train_device, event_nums=cfg.test_events_nums)
            print ("Train Accuracy {}".format(train_accuracy))
            test_loss= test_step(test_loader, train_device)
            print ("Test Loss {}".format(test_loss))
            fout.write("%f,"%train_accuracy)        
            fout.write("%f,"%test_accuracy)
            fout.write("%f,"%loss)
            fout.write("%f,"%test_loss)
            fout.write("%f,"%epoch)
            fout.write("%f"%step)

            fout.write("\n")
        step+=1
fout.close()
