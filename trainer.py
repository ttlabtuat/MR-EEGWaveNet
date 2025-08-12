import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
from  utils import *


class Trainer:
    def __init__(self, model, class_ratio, num_classes):
        self.Model = model
        self.class_ratio = class_ratio
        self.compiled = False
        self.tracker = {'train_tracker': [], 'val_tracker': []}

        # Compute class weights based on class_ratio
        total = class_ratio + 1.0
        class_counts = np.array([class_ratio, 1.0])
        weights = total / (num_classes * class_counts)
        weights = torch.FloatTensor(weights)

        if torch.cuda.is_available():
            weights = weights.cuda()
        self.loss_func = CrossEntropyLoss(weight=weights)
        print(f"Class weights: {weights.cpu().numpy()}")

    def compile(self, learning_rate):
        self.optimizer = Adam(self.Model.parameters(), lr=learning_rate)
        self.lr = learning_rate
        self.compiled = True


    def train(self, train_sz, train_ns, epochs, batch_size=32, patience=30, tr_val_ratio=0.7, directory='model.pt', loss_dir='loss.csv'):
    
        if not self.compiled:
            raise Exception("You need to compile the optimizer before training.")
    
        best_model = copy.deepcopy(self.Model)
        wait = 0
        results = {'train_loss': [], 'val_loss': []}
        train_loss_tracker = []
        val_loss_tracker = []
    
        if torch.cuda.is_available():
            self.Model.cuda()

        loss_amplifier = 10000
        
        for epoch in range(1, epochs + 1): 
    
            self.optimizer = Adam(self.Model.parameters(), lr=self.lr)
            
            # --- Create epoch dataset ---
            full_dataset = create_epoch_dataset(train_sz, train_ns, class_ratio=self.class_ratio)
            (train_data, train_label), (val_data, val_label) = split_dataset(dataset=full_dataset, train_size=tr_val_ratio)
    
            train_loader = torch.utils.data.DataLoader(
                Dataset_tensor(train_data, train_label),
                batch_size=batch_size, 
                shuffle=True, pin_memory=True, num_workers=8
            )
            
            val_loader = torch.utils.data.DataLoader(
                Dataset_tensor(val_data, val_label),
                batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8
            )
    
            # --- Training ---
            self.Model.train()
            train_loss_sum = 0.0
            total_train_samples = 0
    
            for data, target in tqdm(train_loader, desc=f"Train Epoch {epoch}/{epochs}"):
                data = data.float().cuda() if torch.cuda.is_available() else data.float()
                target = target.cuda() if torch.cuda.is_available() else target
    
                self.optimizer.zero_grad()
                pred, _, _ = self.Model(data)
                loss = self.loss_func(pred, target)
                loss.backward()
                self.optimizer.step()
    
                train_loss_sum += loss.item() * len(target)
                total_train_samples += len(target)
    
            final_train_loss = (train_loss_sum / total_train_samples) * loss_amplifier
            
            # --- Validation ---
            self.Model.eval()
            val_loss_sum = 0.0
            total_val_samples = 0
    
            with torch.no_grad():
                for data, target in tqdm(val_loader, desc=f"Valid Epoch {epoch}/{epochs}"):
                    data = data.float().cuda() if torch.cuda.is_available() else data.float()
                    target = target.cuda() if torch.cuda.is_available() else target
    
                    pred, _, _ = self.Model(data)
                    loss = self.loss_func(pred, target)
    
                    val_loss_sum += loss.item() * len(target)
                    total_val_samples += len(target)
    
            final_val_loss = (val_loss_sum / total_val_samples) * loss_amplifier
    
            # --- save Loss information---
            train_loss_tracker.append(final_train_loss)
            val_loss_tracker.append(final_val_loss)
            results['train_loss'].append(round(final_train_loss, 3))
            results['val_loss'].append(round(final_val_loss, 3))
            pd.DataFrame(results, index=range(1, epoch + 1)).to_csv(loss_dir, index_label='epoch')
    
            print(f"Epoch {epoch:03d} | Train Loss: {final_train_loss:.4f} | Val Loss: {final_val_loss:.4f}")
    
            # --- Model saving & early stopping ---
            if epoch > patience:
                if final_val_loss <= min(val_loss_tracker):
                    best_model = copy.deepcopy(self.Model)
                    torch.save(self.Model.state_dict(), directory)
                    wait = 0
                else:
                    wait += 1
            else:
                torch.save(self.Model.state_dict(), directory)
    
            if wait >= patience:
                print("Early stopping triggered.")
                break
    
        self.tracker['train_tracker'] = train_loss_tracker
        self.tracker['val_tracker'] = val_loss_tracker
        self.Model = copy.deepcopy(best_model)
        torch.save(self.Model.state_dict(), directory)
    
        return self.tracker


    def predict(self, test_loader):
        PRED, FEAT, PROB, TARGET = [], [], [], []
    
        print("# EDF Predict ------------------------------------------------")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Model.to(device)
    
        self.Model.eval()
        with torch.no_grad():
            for data, target in tqdm(test_loader):
                data = data.float().cuda() if torch.cuda.is_available() else data.float()
                target = target.cuda() if torch.cuda.is_available() else target
    
                pred, feat, prob = self.Model(data)
                pred = torch.argmax(pred, dim=1).cpu().numpy()
    
                PRED.extend(pred)
                FEAT.append(feat.cpu())
                PROB.append(prob.cpu())
                TARGET.append(target.cpu())
    
        return (
                np.hstack(PRED),
                np.vstack(FEAT),
                np.vstack(PROB),
                np.hstack(TARGET)
                )

