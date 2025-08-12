import torch
from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, n_chans, n_classes, feature_dim, Fs, SEC, tot_feat_len):
        super(Model, self).__init__()

        self.Fs = Fs  # Sampling frequency
        self.tot_feat_len = tot_feat_len  # Total feature length (after all feature concatenations)
        self.SEC = SEC    # List of segment durations (in seconds) for multi-resolution processing

        # Depthwise temporal convolution layers to reduce temporal resolution
        self.temp_conv1 = nn.Conv1d(n_chans, n_chans, kernel_size=2, stride=2 ,groups=n_chans)
        self.temp_conv2 = nn.Conv1d(n_chans, n_chans, kernel_size=2, stride=2 ,groups=n_chans)
        self.temp_conv3 = nn.Conv1d(n_chans, n_chans, kernel_size=2, stride=2 ,groups=n_chans)
        self.temp_conv4 = nn.Conv1d(n_chans, n_chans, kernel_size=2, stride=2 ,groups=n_chans)
        self.temp_conv5 = nn.Conv1d(n_chans, n_chans, kernel_size=2, stride=2 ,groups=n_chans)
        self.temp_conv6 = nn.Conv1d(n_chans, n_chans, kernel_size=2, stride=2 ,groups=n_chans)


        # Channel-wise convolution block 1
        self.chpool1    = nn.Sequential(
                                        nn.Conv1d(n_chans, 32, kernel_size=4,groups=1),
                                        nn.BatchNorm1d(32),
                                        nn.LeakyReLU(0.01),
                                        nn.Conv1d(32, 32, kernel_size=4,groups=1),
                                        nn.BatchNorm1d(32),
                                        nn.LeakyReLU(0.01))

        # Channel-wise convolution block 2
        self.chpool2    = nn.Sequential(
                                        nn.Conv1d(n_chans, 32, kernel_size=4,groups=1),
                                        nn.BatchNorm1d(32),
                                        nn.LeakyReLU(0.01),
                                        nn.Conv1d(32, 32, kernel_size=4,groups=1),
                                        nn.BatchNorm1d(32),
                                        nn.LeakyReLU(0.01))

        # Channel-wise convolution block 3
        self.chpool3    = nn.Sequential(
                                        nn.Conv1d(n_chans, 32, kernel_size=4,groups=1),
                                        nn.BatchNorm1d(32),
                                        nn.LeakyReLU(0.01),
                                        nn.Conv1d(32, 32, kernel_size=4,groups=1),
                                        nn.BatchNorm1d(32),
                                        nn.LeakyReLU(0.01))

        # Channel-wise convolution block 4
        self.chpool4    = nn.Sequential(
                                        nn.Conv1d(n_chans, 32, kernel_size=4,groups=1),
                                        nn.BatchNorm1d(32),
                                        nn.LeakyReLU(0.01),
                                        nn.Conv1d(32, 32, kernel_size=4,groups=1),
                                        nn.BatchNorm1d(32),
                                        nn.LeakyReLU(0.01))

        # Channel-wise convolution block 5
        self.chpool5    = nn.Sequential(
                                        nn.Conv1d(n_chans, 32, kernel_size=4,groups=1),
                                        nn.BatchNorm1d(32),
                                        nn.LeakyReLU(0.01),
                                        nn.Conv1d(32, 32, kernel_size=4,groups=1),
                                        nn.BatchNorm1d(32),
                                        nn.LeakyReLU(0.01))

        # Linear projection of concatenated features to embedding space
        self.cal_feature = nn.Sequential(
                                        nn.Linear(160,64),
                                        nn.LeakyReLU(0.01),
                                        nn.Linear(64,feature_dim))


        # Classifier head that takes concatenated features and outputs logits
        self.classifier = nn.Sequential(
                                        nn.Linear(self.tot_feat_len,64),
                                        nn.LeakyReLU(0.01),
                                        nn.Linear(64,32),
                                        nn.Sigmoid(),
                                        nn.Linear(32,n_classes))


    def feature(self, x):
        # Apply stacked depthwise temporal convolutions
        temp_x  = self.temp_conv1(x)               
        temp_w1 = self.temp_conv2(temp_x)         
        temp_w2 = self.temp_conv3(temp_w1)      
        temp_w3 = self.temp_conv4(temp_w2)       
        temp_w4 = self.temp_conv5(temp_w3)      
        temp_w5 = self.temp_conv6(temp_w4) 

        
        # Apply channel-wise convolutional blocks and average over time dimension
        w1      = self.chpool1(temp_w1).mean(dim=(-1))
        w2      = self.chpool2(temp_w2).mean(dim=(-1))
        w3      = self.chpool3(temp_w3).mean(dim=(-1))
        w4      = self.chpool4(temp_w4).mean(dim=(-1))
        w5      = self.chpool5(temp_w5).mean(dim=(-1))

        
        # Concatenate features from multiple resolutions
        concat_vector  = torch.cat([w1,w2,w3,w4,w5],1) 
        
        # Project to lower-dimensional normalized feature vector
        feature  = self.cal_feature(concat_vector)

        return F.normalize(feature, dim=-1)


    def forward(self, x, training=True):
        # Total number of time samples in the input
        win_sample = x.shape[-1]

        # Iterate over different temporal resolutions
        for k, sec in enumerate(self.SEC):
            samples = int(sec*self.Fs) # convert duration (sec) to samples
            num_win = win_sample//samples # number of windows for this duration

            for i in range(num_win):
                y = x[:,:, samples*i:samples*(i+1)]   # extract window
                feat = self.feature(y)   # extract feature for the window

                if k==0:
                    Feature = feat.clone() # initialize
                else:     
                    Feature = torch.cat([Feature, feat], dim=1) # concatenate along feature dimension

        # Compute class scores and softmax probability
        classes     = nn.functional.log_softmax(self.classifier(Feature),dim=1)
        probability = nn.functional.softmax(self.classifier(Feature),dim=1) 


        # Return log-probabilities, features, and probabilities
        return classes, F.normalize(Feature, dim=-1), probability
