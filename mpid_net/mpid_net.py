###
### Note11-08-2019, using PyToch_v1.0.1, one has to use GroupNorm
### to avoid bug (we believe in PyTorch) in BatchNorm layer during inference.
###

import torch
import torch.nn as nn

class MPID(nn.Module):
    # eps: default value for batchnorm
    def __init__(self, dropout=0.5, num_classes=cfg.num_class, eps = 1e-05, running_stats=False):
        super(MPID, self).__init__()

        self.features = nn.Sequential(
            #layer 1, 1_0 with stride = 2, others = 1
            #each sublayer has an active function of ReLU except 5_2
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Defualt setup for batch norma
            #nn.BatchNorm2d(64,eps=eps, track_running_stats=running_stats),
            nn.GroupNorm(64,64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            #nn.BatchNorm2d(64,eps=eps, track_running_stats=running_stats),
            nn.GroupNorm(64,64),
            nn.AvgPool2d(2, padding=1),
            #layer 2
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            #nn.BatchNorm2d(96,eps=eps, track_running_stats=running_stats),
            nn.GroupNorm(96,96),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1),
            nn.ReLU(),
            #nn.BatchNorm2d(96,eps=eps, track_running_stats=running_stats),
            nn.GroupNorm(96,96),
            nn.AvgPool2d(2, padding=1),
            #layer 3
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            #nn.BatchNorm2d(128,eps=eps, track_running_stats=running_stats),
            nn.GroupNorm(128,128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            nn.ReLU(),
            #nn.BatchNorm2d(128,eps=eps, track_running_stats=running_stats),
            nn.GroupNorm(128,128),
            nn.AvgPool2d(2, padding=1),
            #layer 4
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            #nn.BatchNorm2d(160,eps=eps, track_running_stats=running_stats),
            nn.GroupNorm(160,160),
            nn.Conv2d(in_channels=160, out_channels=160, kernel_size=3, stride=1),
            nn.ReLU(),
            #nn.BatchNorm2d(160,eps=eps, track_running_stats=running_stats),
            nn.GroupNorm(160,160),
            nn.AvgPool2d(2, padding=1),
            #layer 5
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            #nn.BatchNorm2d(192,eps=eps, track_running_stats=running_stats),
            nn.GroupNorm(192,192),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1),
            #nn.ReLU(),
            #nn.BatchNorm2d(192,eps=eps, track_running_stats=running_stats),
            nn.GroupNorm(192,192),
            nn.AvgPool2d(2, padding=1)
        )
        
        # Global Average Pooling
        #self.avgpool = nn.AdaptiveAvgPool2d((8,8))

        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(192 * 8 * 8, 192 * 8),
            nn.Dropout(dropout),
            nn.Linear(192*8, 192),
            nn.Linear(192, num_classes),
        )
        
    def forward(self, x):
        x=self.features(x)
        #x=self.avgpool(x)
        x=torch.flatten(x, 1)
        x=self.dropout(x)
        x=self.classifier(x)
        return x


if __name__ == '__main__':
    x = torch.ones([512, 512])
    mpid = MPID()
    print (mpid)
    print ("mpid.training, ",mpid.training)
    #mpid.forward()
    print (mpid(x.view((-1,1,512,512))))
