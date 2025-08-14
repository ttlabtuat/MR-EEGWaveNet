## Model Training and Validation Strategy
The models were evaluated using a Leave-One-Subject-Out (LOSO) cross-validation approach.   
In each fold, one patient was reserved for testing, while the remaining patients were used for training, as illustrated below: 
<p align="center"> 
<img width="500" height="500" alt="LOSO" src="https://github.com/user-attachments/assets/023b3436-f601-4c4f-a823-fb58eb05e6a5" />
</p>

## Dataset Preparation for Training  
To address the data imbalance problem between seizure and nonseizure EEG segments, we adopted the following two strategies:

I. Seizure Data – Oversampling
Consecutive seizure segments (as per expert annotations) were overlapped by 80% to generate more training examples.

II. Nonseizure Data – Undersampling
Nonseizure segments were randomly selected from interictal EEG recordings, using non-overlapping windows.

The ratio between seizure and nonseizure samples was maintained at 1:2.


## Test Set Segmentation

For the held-out patient in each LOSO fold:  
EEG recordings were segmented using a non-overlapping sliding window with a fixed window length.

Each segment was assigned a label:  
  0 → Nonseizure (negative class)  
  1 → Seizure (positive class) based on ground-truth expert annotations.
