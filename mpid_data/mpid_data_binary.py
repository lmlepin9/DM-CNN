import numpy as np
import random

import h5py

import torch
from torch.utils.data import Dataset

def image_modify(img):
    img_mod = np.where(img<10,    0,img)
    img_mod = np.where(img>500, 500,img_mod)
    return img_mod

def EvDisp(input_file):
    producer = 'image2d_binary'
    f = h5py.File(input_file,'r')
    event_id_list = f['eventid']
    wire_set = f['image2d'][producer]
    image=wire_set[list(wire_set.keys())[0]]
    return event_id_list, image

#Plane 2 is the only one present in the cropped dataset, therefore we use plane = 0 
class MPID_Dataset(Dataset):
    def __init__(self, input_file, image_tree, device, plane=0,augment=False, verbose=False):
        self.plane=plane
        self.augment=augment
        self.verbose=verbose
         
        self.input_file = input_file
        #f = h5py.File(input_file,'r')
        #self.particle_image_chain = f['image2d']['image2d_binary_tree']

        if (device):
            self.device=device
        else:
            self.device="cpu"
        
    def __getitem__(self, ENTRY):
        # Reading Image

        #self.particle_image_chain.GetEntry(ENTRY)
        #self.this_image_cpp_object = self.particle_image_chain.image2d_image2d_binary_branch
        
        self.event_info = [[],[],[]]

        self.event_id_list, self.this_image = EvDisp(self.input_file)

        for element in self.event_id_list:
            self.event_info[0] = element[1] # run
            self.event_info[1] = element[2] # subrun
            self.event_info[2] = element[3] # event

        # Image Thresholding
        #self.this_image=image_modify(self.this_image)
        
        if self.augment:
            if random.randint(0, 1):
                self.this_image = np.fliplr(self.this_image)
            if random.randint(0, 1):
                self.this_image = self.this_image.transpose(1,0)
        #self.this_image = torch.from_numpy(self.this_image.copy())

        #self.this_image=self.this_image.clone().detach()
        
        # Creating labels 
        self.event_label = torch.zeros([2])
        # Signal events are labeled with run = 100 
        if(self.event_info[0] == 100):
            self.event_label[0] = 1
        
        else:
            self.event_label[1] = 1
                
                              
        return (self.this_image, self.event_label, self.event_info)
        #return (self.this_image, self.event_label)

    #def __len__(self):
        #return self.particle_image_chain.GetEntries()

train_file = "/Users/juliamaidannyk/Downloads/dark_trident_train_overlay_larcv_cropped.h5"
train_data = MPID_Dataset(train_file, "image2d_image2d_binary_tree", device='cpu', plane=0, augment=False)
