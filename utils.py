import numpy as np
import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split

from sklearn.metrics import ( accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, roc_auc_score
                            )

class Dataset_tensor(Dataset):
    def __init__(self, data, label):
        self.data  = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data_tensor  = torch.tensor(self.data[idx], dtype=torch.float32)
        label_tensor = torch.tensor(int(self.label[idx]), dtype=torch.long)
        return data_tensor, label_tensor



def create_epoch_dataset(train_sz, train_ns, class_ratio):
    N = train_sz.shape[0]
    S = int(class_ratio*N)

    # Select 2N random non-seizure samples
    indices = np.random.choice(train_ns.shape[0], size=S, replace=False)
    sel_train_ns = train_ns[indices]  # (2N, C, T)

    # Combine seizure and non-seizure data
    data = np.concatenate([train_sz, sel_train_ns], axis=0)  # (3N, C, T)

    # Create labels: 1 for seizure, 0 for non-seizure
    label = np.concatenate([np.ones(N), np.zeros(S)], axis=0)  # (3N,)

    return Dataset_tensor(data, label)




def split_dataset(dataset, train_size=0.7):
    """
    Splits the dataset into training and validation sets.

    Args:
        dataset (Dataset_tensor): Custom dataset object.
        train_size (float): Proportion of the dataset to include in the train split (default: 0.7).

    Returns:
        (train_data, train_label), (val_data, val_label): Tuple of NumPy arrays.
    """
    # Extract data and labels
    data  = dataset.data
    label = dataset.label

    # Perform stratified split
    train_data, val_data, train_label, val_label = train_test_split(
        data, label,
        train_size=train_size,
        stratify=label
    )

    return (train_data, train_label), (val_data, val_label)




def predict2perf(y_pred, y_true, prob):
    # Convert to numpy arrays in case they're not
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Per-class Precision, Recall, F1
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall    = recall_score(y_true, y_pred, average=None)
    f1        = f1_score(y_true, y_pred, average=None)

    # Macro F1
    macro_f1  = f1_score(y_true, y_pred, average='macro')

    # Specificity
    conf_matrix = confusion_matrix(y_true, y_pred)
    specificity = []
    for i in range(len(conf_matrix)):
        tn = np.sum(conf_matrix) - np.sum(conf_matrix[i, :]) - np.sum(conf_matrix[:, i]) + conf_matrix[i, i]
        fp = np.sum(conf_matrix[:, i]) - conf_matrix[i, i]
        specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    specificity = np.array(specificity, dtype=np.float32)

    # ROC-AUC for class 1 (seizure)
    try:
        roc_auc = roc_auc_score(y_true, prob[:, 1])
    except ValueError:
        roc_auc = np.nan  # in case only one class exists in y_true

    # Cast all outputs to float32
    accuracy    = np.float32(accuracy)
    macro_f1    = np.float32(macro_f1)
    roc_auc     = np.float32(roc_auc)

    precision   = precision.astype(np.float32)
    recall      = recall.astype(np.float32)
    f1          = f1.astype(np.float32)
    specificity = specificity.astype(np.float32)

    cs = 1
    return accuracy, precision[cs], recall[cs], specificity[cs], f1[cs], macro_f1, roc_auc
