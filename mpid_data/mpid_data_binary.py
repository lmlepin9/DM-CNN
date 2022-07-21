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
            self.event_info[0] = element[ENTRY][3] # run
            self.event_info[1] = element[ENTRY][2] # subrun
            self.event_info[2] = element[ENTRY][1] # event

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

    def __len__(self):
        f = h5py.File(self.input_file,'r')
        return len(f['eventid'])
