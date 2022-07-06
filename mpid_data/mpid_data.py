import numpy as np
from larcv import larcv
import random

import ROOT
from ROOT import TChain

import torch
from torch.utils.data import Dataset

def image_modify(img):
    img_mod = np.where(img<10,    0,img)
    img_mod = np.where(img>500, 500,img_mod)
    return img_mod

class MPID_Dataset(Dataset):
    def __init__(self, input_file, image_tree, device, nclasses, plane=0, augment=False, verbose=False):
        self.plane=plane
        self.augment=augment
        self.verbose=verbose
        self.nclasses=nclasses
        #self.particle_mctruth_chain = TChain(mctruth_tree)
        #self.particle_mctruth_chain.AddFile(input_file)
        #print('Found',self.particle_mctruth_chain.GetEntries(),'entries!')
        self.particle_image_chain = TChain(image_tree)
        self.particle_image_chain.AddFile(input_file)
        if (device):
            self.device=device
        else:
            self.device="cpu"
        
    def __getitem__(self, ENTRY):
        # Reading Image

        #print ("open ENTRY @ {}".format(ENTRY))

        self.particle_image_chain.GetEntry(ENTRY)
        self.this_image_cpp_object = self.particle_image_chain.image2d_image2d_branch
        # self.this_image_cpp_object is <ROOT.larcv::EventImage2D object at 0xb5be5a0>
        self.this_image=larcv.as_ndarray(self.this_image_cpp_object.as_vector()[self.plane])
        # Image Thresholding
        self.this_image=image_modify(self.this_image)

        #print (self.this_image).
        #print ("sum, ")
        #if (np.sum(self.this_image) < 9000):
        #    ENTRY+
        
        if self.augment:
            if random.randint(0, 1):
            #if True:
                #if (self.verbose): print ("flipped")
                self.this_image = np.fliplr(self.this_image)
            if random.randint(0, 1):
            #if True:
                #if (self.verbose): print ("transposed")
                self.this_image = self.this_image.transpose(1,0)
        self.this_image = torch.from_numpy(self.this_image.copy())
        #self.this_image=torch.tensor(self.this_image, device=self.device).float()

        self.this_image=self.this_image.clone().detach()
        
        # Reading Truth Info
        #self.particle_mctruth_chain.GetEntry(ENTRY)
        #self.this_mctruth_cpp_object = self.particle_mctruth_chain.particle_mctruth_branch
        #self.this_mctruth = torch.zeros([5])

        self.event_label = torch.zeros([self.nclasses])

        if (self.this_image_cpp_object.subrun()==11): 
            self.event_label[0]=1
        if (self.this_image_cpp_object.subrun()==22):
            self.event_label[1]=1
        if (self.this_image_cpp_object.subrun()==13):
            self.event_label[2]=1
        else:
            pass


        # Print out particle information
        #particle_chain = ROOT.TChain("particle_mctruth_tree")
        #particle_chain.GetEntry(ENTRY)
        #entry_data = particle_chain.particle_mctruth_branch
        """particle_array = self.this_mctruth_cpp_object.as_vector()
        engs = []
        for index, particle in enumerate(particle_array):
            engs.append(particle.energy_init)"""
                                
        return (self.this_image, self.event_label,
                self.this_image_cpp_object.run(), self.this_image_cpp_object.subrun(), self.this_image_cpp_object.event())

    def __len__(self):
        #assert self.particle_image_chain.GetEntries()== self.particle_mctruth_chain.GetEntries()
        return self.particle_image_chain.GetEntries()

