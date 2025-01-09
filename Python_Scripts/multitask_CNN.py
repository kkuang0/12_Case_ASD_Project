import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
import os
from pathlib import Path
import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torchvision
import io
from tqdm import tqdm  # <-- Added import

class AxonDataset(Dataset):
    def __init__(self, metadata_df, base_dir, transform=None, augment=None):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing image metadata
            base_dir (str): Base directory containing the Images folder
            transform (callable, optional): Base transforms
            augment (callable, optional): Augmentation transforms
        """
        self.metadata = metadata_df
        self.base_dir = Path(base_dir)
        self.transform = transform
        self.augment = augment
        
        # Create label encoders
        self.condition_map = {'CTR': 0, 'ASD': 1}
        self.region_map = {'A25': 0, 'A46': 1, 'OFC': 2}
        self.depth_map = {'DWM': 0, 'SWM': 1}
        
    def __len__(self):
        """
        Required for DataLoader to know the size of the dataset.
        """
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # Get image path and labels
        row = self.metadata.iloc[idx]
        # Prepend 'Images' to the path
        img_path = self.base_dir / 'Images' / row['image_path']
        
        # Load image
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        # Apply augmentation if specified
        if self.augment is not None:
            image = self.augment(image)
        
        # Apply base transforms
        if self.transform is not None:
            image = self.transform(image)
        
        # Get labels
        condition_label = self.condition_map[row['condition']]
        region_label = self.region_map[row['region']]
        depth_label = self.depth_map[row['depth']]
        
        return {
            'image': image,
            'condition': condition_label,
            'region': region_label,
            'depth': depth_label,
            'subject_id': row['subject_id']
        }

class MultiTaskCNN(nn.Module):
    def __init__(self):
        super(MultiTaskCNN, self).__init__()
        
        # Shared convolutional layers
        self.shared_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )
        
        # Task-specific heads
        self.condition_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # ASD vs CTR
        )
        
        self.region_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 3)  # A25, A46, OFC
        )
        
        self.depth_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # DWM vs SWM
        )
        
    def forward(self, x):
        shared_features = self.shared_conv(x)
        
        return (
            self.condition_head(shared_features),
            self.region_head(shared_features),
            self.depth_head(shared_features)
        )

class MultiTaskLoss:
    def __init__(self, weights={'condition': 1.0, 'region': 1.0, 'depth': 1.0}):
        self.weights = weights
        self.criterion = nn.CrossEntropyLoss()
        
    def __call__(self, predictions, targets):
        """
        predictions: dict or tuple of (condition_pred, region_pred, depth_pred)
        targets: dict of { 'condition': ..., 'region': ..., 'depth': ... }
        """
        losses = {}
        total_loss = 0
        
        for task, weight in self.weights.items():
            losses[task] = self.criterion(predictions[task], targets[task])
            total_loss += weight * losses[task]
            
        return total_loss, losses

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, 
                 device, num_epochs=100, early_stopping_patience=10,
                 exp_name=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.patience = early_stopping_patience
        
        # Initialize TensorBoard writer
        if exp_name is None:
            exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(f'runs/{exp_name}')
        
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': {}, 'val_acc': {}
        }
        
    def log_images(self, images, epoch):
        """Log a batch of images to TensorBoard."""
        img_grid = torchvision.utils.make_grid(images, normalize=True)
        self.writer.add_image('Sample Images', img_grid, epoch)
    
    def log_confusion_matrices(self, epoch):
        self.model.eval()
        predictions = {
            'condition': [], 'region': [], 'depth': []
        }
        targets = {
            'condition': [], 'region': [], 'depth': []
        }
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                condition_pred, region_pred, depth_pred = self.model(images)
                outputs = {
                    'condition': condition_pred,
                    'region': region_pred,
                    'depth': depth_pred
                }
                
                for task in ['condition', 'region', 'depth']:
                    pred = torch.argmax(outputs[task], dim=1).cpu().numpy()
                    target = batch[task].cpu().numpy()
                    predictions[task].extend(pred)
                    targets[task].extend(target)
        
        # Create and log confusion matrices
        for task in ['condition', 'region', 'depth']:
            cm = confusion_matrix(targets[task], predictions[task])
            fig, ax = plt.subplots(figsize=(8, 8))
            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
            ax.set_title(f'{task.capitalize()} Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            
            # Convert plot to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)
            img = transforms.ToTensor()(img)
            
            self.writer.add_image(f'{task}/confusion_matrix', img, epoch)
            plt.close()
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        task_correct = {'condition': 0, 'region': 0, 'depth': 0}
        task_total = {'condition': 0, 'region': 0, 'depth': 0}
        task_losses = {'condition': 0, 'region': 0, 'depth': 0}
        
        # Wrap train_loader with tqdm
        for batch_idx, batch in enumerate(
            tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}", leave=False)
        ):
            images = batch['image'].to(self.device)
            targets = {
                task: batch[task].to(self.device)
                for task in ['condition', 'region', 'depth']
            }
            
            # Log the first batch of the very first epoch
            if epoch == 0 and batch_idx == 0:
                self.log_images(images, epoch)
            
            self.optimizer.zero_grad()
            
            # Get predictions
            condition_pred, region_pred, depth_pred = self.model(images)
            predictions = {
                'condition': condition_pred,
                'region': region_pred,
                'depth': depth_pred
            }
            
            # Calculate loss
            loss, individual_losses = self.criterion(predictions, targets)
            
            loss.backward()
            self.optimizer.step()
            
            # Record losses and accuracies
            total_loss += loss.item()
            for task, task_loss in individual_losses.items():
                task_losses[task] += task_loss.item()
            
            # Calculate accuracy
            for task in ['condition', 'region', 'depth']:
                pred = torch.argmax(predictions[task], dim=1)
                task_correct[task] += (pred == targets[task]).sum().item()
                task_total[task] += targets[task].size(0)
            
            # Log batch-level metrics
            step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Batch/Loss', loss.item(), step)
        
        # Calculate average loss and accuracies
        avg_loss = total_loss / len(self.train_loader)
        accuracies = {
            task: task_correct[task] / task_total[task]
            for task in task_correct.keys()
        }
        avg_task_losses = {
            task: loss_val / len(self.train_loader)
            for task, loss_val in task_losses.items()
        }
        
        # Log epoch-level metrics
        self.writer.add_scalar('Loss/train', avg_loss, epoch)
        for task in accuracies.keys():
            self.writer.add_scalar(f'Accuracy/train/{task}', accuracies[task], epoch)
            self.writer.add_scalar(f'Loss/train/{task}', avg_task_losses[task], epoch)
        
        return avg_loss, accuracies
    
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        task_correct = {'condition': 0, 'region': 0, 'depth': 0}
        task_total = {'condition': 0, 'region': 0, 'depth': 0}
        task_losses = {'condition': 0, 'region': 0, 'depth': 0}
        
        # Wrap val_loader with tqdm
        for batch_idx, batch in enumerate(
            tqdm(self.val_loader, desc=f"Validation Epoch {epoch+1}", leave=False)
        ):
            with torch.no_grad():
                images = batch['image'].to(self.device)
                targets = {
                    task: batch[task].to(self.device)
                    for task in ['condition', 'region', 'depth']
                }
                
                # Get predictions
                condition_pred, region_pred, depth_pred = self.model(images)
                predictions = {
                    'condition': condition_pred,
                    'region': region_pred,
                    'depth': depth_pred
                }
                
                loss, individual_losses = self.criterion(predictions, targets)
                
                # Record losses and accuracies
                total_loss += loss.item()
                for task, task_loss in individual_losses.items():
                    task_losses[task] += task_loss.item()
                
                # Calculate accuracy
                for task in ['condition', 'region', 'depth']:
                    pred = torch.argmax(predictions[task], dim=1)
                    task_correct[task] += (pred == targets[task]).sum().item()
                    task_total[task] += targets[task].size(0)
        
        # Calculate average loss and accuracies
        avg_loss = total_loss / len(self.val_loader)
        accuracies = {
            task: task_correct[task] / task_total[task]
            for task in task_correct.keys()
        }
        avg_task_losses = {
            task: loss_val / len(self.val_loader)
            for task, loss_val in task_losses.items()
        }
        
        # Log metrics
        self.writer.add_scalar('Loss/val', avg_loss, epoch)
        for task in accuracies.keys():
            self.writer.add_scalar(f'Accuracy/val/{task}', accuracies[task], epoch)
            self.writer.add_scalar(f'Loss/val/{task}', avg_task_losses[task], epoch)
        
        return avg_loss, accuracies
    
    def train(self):
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Log model graph. If you need to allow dict-like output, set use_strict_trace=False
        sample_input = next(iter(self.train_loader))['image'].to(self.device)
        try:
            self.writer.add_graph(self.model, sample_input, use_strict_trace=False)
        except Exception as e:
            print(f"Warning: Could not add model graph to TensorBoard: {str(e)}")
        
        for epoch in range(self.num_epochs):
            # Training
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_acc = self.validate(epoch)
            
            # Log confusion matrices every 5 epochs
            if epoch % 5 == 0:
                self.log_confusion_matrices(epoch)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            for task in train_acc.keys():
                if task not in self.history['train_acc']:
                    self.history['train_acc'][task] = []
                    self.history['val_acc'][task] = []
                self.history['train_acc'][task].append(train_acc[task])
                self.history['val_acc'][task].append(val_acc[task])
            
            # Print progress
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            for task in train_acc.keys():
                print(f"{task.capitalize()} - Train Acc: {train_acc[task]:.4f}, "
                      f"Val Acc: {val_acc[task]:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss,
                }, 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("\nEarly stopping triggered")
                    break
        
        self.writer.close()
        return self.history

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Set base directory (this should be the directory containing the Images folder)
    base_dir = Path("C:/Kelvin_ASD_v2")  # Adjust this to your actual path
    
    # Load metadata
    metadata_df = pd.read_csv('metadata/dataset_metadata.csv')
    
    # Setup transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    # Create datasets
    train_data = metadata_df[metadata_df['split'] == 'train']
    val_data = metadata_df[metadata_df['split'] == 'test']
    
    # Pass the base directory to the datasets
    train_dataset = AxonDataset(train_data, base_dir, transform=transform)
    val_dataset = AxonDataset(val_data, base_dir, transform=transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = MultiTaskCNN().to(device)
    
    # Setup loss and optimizer
    criterion = MultiTaskLoss(weights={'condition': 1.0, 'region': 1.0, 'depth': 1.0})
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create experiment name
    exp_name = f"multitask_cnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create trainer and train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=100,
        early_stopping_patience=10,
        exp_name=exp_name
    )
    
    # Train model
    history = trainer.train()

if __name__ == "__main__":
    main()
