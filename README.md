# MR-EEGWaveNet: Multiresolutional EEGWaveNet for Seizure Detection from Long EEG Recordings
Multiresolutional EEGWaveNet (MR-EEGWaveNet) is a novel end-to-end deep learning model for automatic seizure detection from EEG signals. This model is designed to effectively differentiate seizure events from background EEG and artifacts/noise by capturing both temporal dependencies and spatial relationships.<br>


## Key Features
**_Multiresolutional Architecture:_** Captures temporal dependencies across different time scales.<br>
**_Spatio-Temporal Convolution:_** Enhances learning of inter-channel spatial patterns and intra-channel temporal dynamics.<br>

## Modular Design 
**_Convolution Module:_** Extracts features using depth-wise and spatio-temporal convolution. <br>
**_Feature Extraction Module:_** Reduces dimensionality of features from EEG segments and sub-segments.<br>
**_Predictor Module:_** Classifies concatenated features using a fully connected network.<br>

<!-- This section explains the model architecture -->
## A) Convolution module 
<p align="center"> 
<img width="800" height="560" alt="conv_module" src="https://github.com/user-attachments/assets/d9a32bbc-8905-4d5a-bb8f-9617c254e9c4" />
</p>
      
## MR-EEGWaveNet architecture (Example)
<p align="center"> 
<img width="1000" height="700" alt="mr-eegwavenet-arch-image" src="https://github.com/user-attachments/assets/01c94c8e-58f7-4f19-a3cf-2fd47e100ec0" />
</p>


## Evaluation: Leave-One-Subject-Out (LOSO) cross-validation scheme
<p align="center"> 
<img width="500" height="500" alt="LOSO" src="https://github.com/user-attachments/assets/023b3436-f601-4c4f-a823-fb58eb05e6a5" />
</p>

## Versions used in the Experiment
```
Python 3.9.13
PyTorch 2.7.1
CUDA 12.6 
NumPy 1.24.4 
Scikit-learn 1.0.2
MNE 1.8.0
Pandas 1.4.4
```
## Installation
pip install pyod     # for anomaly detection


## Example of Model usage
```
import torch
import numpy as np
from mreegwavenet import Model  # Make sure the class above is saved in mreegwavenet.py


# 1. Define model parameters (must match those used during training)
n_chans = 19           # EEG channels (example)
n_classes = 2          # Binary classification
feature_dim = 32       # Embedding dimension
Fs = 500               # Sampling rate in Hz
MP = [10.0, 2.0]       # Multi-resolution Parameter (in seconds)


# 2. Initialize the model
model = Model(n_chans, n_classes, feature_dim, Fs, MP)

model_path = "pre-trained-model.pth" # Change to the directory containing the pre-trained model.

# 3. Load the pre-trained weights
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()  # Set to evaluation mode (important)

# 4. Load or prepare your test data (batch_size, n_chans, time_points)
# Example: generate dummy test data with 5 samples
batch_size = 5
time_points = int(MP[0] * Fs)  
x_test = torch.randn(batch_size, n_chans, time_points)

# 5. Run inference (no gradient calculation)
with torch.no_grad():
    predicted_classes, feature, probs = model(x_test, training=False)

# 6. Print results
print("Predicted Classes:", predicted_classes)
print("Probabilities:", probs)
print("Features Shape:", feature.shape)
```



## Citation
If you use this repository, code, or any part of the MR-EEGWaveNet model in your research, please cite the following paper:
```
@misc{hassan2025mreegwavenet,
                  title={MR-EEGWaveNet: Multiresolutional EEGWaveNet for Seizure Detection from Long EEG Recordings},  
                  author={Kazi Mahmudul Hassan and Xuyang Zhao and Hidenori Sugano and Toshihisa Tanaka},  
                  year={2025},   
                  eprint={2505.17972},   
                  archivePrefix={arXiv},    
                  primaryClass={cs.CV},    
                  url={https://arxiv.org/abs/2505.17972},
            }
```


## Contact
For any questions or collaboration opportunities, feel free to open an issue or reach out to hassan@sip.tuat.ac.jp.
