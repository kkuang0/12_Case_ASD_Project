import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard.summary import hparams  
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import random
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io
from tqdm import tqdm  

from PIL import Image, ImageEnhance, ImageFilter

import optuna

from torchvision.models import efficientnet_b2



class AxonAugmenter:
    """Custom augmentation pipeline for axon microscopy images."""
    
    def __init__(self,
                 rotation_range=360,
                 brightness_range=(0.8, 1.2),
                 contrast_range=(0.8, 1.2),
                 noise_range=(0.0, 0.05),
                 blur_prob=0.3,
                 blur_radius=(0.5, 1.5),
                 zoom_range=(0.85, 1.15),
                 flip_prob=0.5):
        """
        Args:
            rotation_range (int): Range for random rotation
            brightness_range (tuple): Range for brightness adjustment
            contrast_range (tuple): Range for contrast adjustment
            noise_range (tuple): Range for noise amplitude
            blur_prob (float): Probability of applying Gaussian blur
            blur_radius (tuple): Range for blur radius
            zoom_range (tuple): Range for random zoom
            flip_prob (float): Probability of horizontal/vertical flip
        """
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_range = noise_range
        self.blur_prob = blur_prob
        self.blur_radius = blur_radius
        self.zoom_range = zoom_range
        self.flip_prob = flip_prob
    
    def add_microscope_noise(self, image):
        """Add realistic microscope noise (Gaussian + Poisson)."""
        image_array = np.array(image)
        
        # Add Gaussian noise
        noise_amplitude = random.uniform(*self.noise_range)
        gaussian_noise = np.random.normal(0, noise_amplitude, image_array.shape)
        
        # Add Poisson noise (simulate photon counting noise)
        poisson_noise = np.random.poisson(image_array).astype(float) - image_array
        poisson_noise *= noise_amplitude
        
        # Combine noises and clip to valid range
        noisy_image = image_array + gaussian_noise + poisson_noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy_image)
    
    def random_blur(self, image):
        """Apply random Gaussian blur."""
        if random.random() < self.blur_prob:
            radius = random.uniform(*self.blur_radius)
            return image.filter(ImageFilter.GaussianBlur(radius))
        return image
    
    def random_zoom(self, image):
        """Apply random zoom while maintaining aspect ratio."""
        zoom_factor = random.uniform(*self.zoom_range)
        
        # Calculate new dimensions
        w, h = image.size
        new_w = int(w * zoom_factor)
        new_h = int(h * zoom_factor)
        
        # Resize and crop/pad to original size
        if zoom_factor > 1:  # Zoom in
            image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
            left = (new_w - w) // 2
            top = (new_h - h) // 2
            image = image.crop((left, top, left + w, top + h))
        else:  # Zoom out
            temp_image = Image.new(image.mode, (w, h), (0,))
            resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
            left = (w - new_w) // 2
            top = (h - new_h) // 2
            temp_image.paste(resized, (left, top))
            image = temp_image
            
        return image
    
    def __call__(self, image):
        """Apply the augmentation pipeline to an image."""
        # Convert to PIL if necessary
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Basic geometric transformations
        if random.random() < self.flip_prob:
            image = TF.hflip(image)
        if random.random() < self.flip_prob:
            image = TF.vflip(image)
            
        # Random rotation
        if self.rotation_range > 0:
            angle = random.uniform(-self.rotation_range/2, self.rotation_range/2)
            image = image.rotate(angle, Image.Resampling.BILINEAR, expand=False)
        
        # Intensity transformations
        if self.brightness_range != (1.0, 1.0):
            brightness_factor = random.uniform(*self.brightness_range)
            image = ImageEnhance.Brightness(image).enhance(brightness_factor)
            
        if self.contrast_range != (1.0, 1.0):
            contrast_factor = random.uniform(*self.contrast_range)
            image = ImageEnhance.Contrast(image).enhance(contrast_factor)
        
        # Microscope-specific augmentations
        image = self.random_blur(image)
        image = self.add_microscope_noise(image)
        image = self.random_zoom(image)
        
        return image

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
        relative_path_str = row['image_path'].replace('\\', os.sep)
        relative_path = Path(relative_path_str)
    
        # Get all parts of the path.
        parts = list(relative_path.parts)
    
        # Replace spaces with underscores in all parts except the last (the filename).
        parts_modified = [part.replace(" ", "_") for part in parts[:-1]] + [parts[-1]]
        
        # Reconstruct the modified relative path.
        modified_relative_path = Path(*parts_modified)
        
        # Prepend 'Images' to the path
        img_path = self.base_dir / 'Images' / modified_relative_path
        
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

class MultiTaskEfficientNet(nn.Module):
    def __init__(self):
        super(MultiTaskEfficientNet, self).__init__()
        
        # Load a pretrained EfficientNetV2-B2 from torchvision
        # Specify weights='IMAGENET1K_V1' or the default for this model:
        self.backbone = efficientnet_v2_b2(weights=torchvision.models.EfficientNet_V2_B2_Weights.IMAGENET1K_V1)

        # Replace final classifier with an identity
        # so we can attach multiple heads ourselves
        self.backbone.classifier = nn.Identity()

        # EfficientNetV2-B2's last feature dimension is typically 1408
        in_features = 1408
        
        # Create multi-task heads. Feel free to adjust hidden dim, dropout, etc.
        self.condition_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # ASD vs CTR
        )
        
        self.region_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 3)  # A25, A46, OFC
        )
        
        self.depth_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # DWM vs SWM
        )
        
        # Modify first convolution to accept 1-channel input instead of 3
        # The first conv is typically self.backbone.features[0][0]
        first_conv = self.backbone.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=(first_conv.bias is not None)
        )
        # Copy mean of existing weights to the new conv
        with torch.no_grad():
            new_conv.weight = nn.Parameter(torch.mean(first_conv.weight, dim=1, keepdim=True))
        if first_conv.bias is not None:
            new_conv.bias = first_conv.bias
        
        self.backbone.features[0][0] = new_conv
        
    def forward(self, x):
        # Extract features using the EfficientNet backbone
        x = self.backbone(x)
        
        # Pass the pooled features to each classification head
        condition_pred = self.condition_head(x)
        region_pred = self.region_head(x)
        depth_pred = self.depth_head(x)
        
        return (condition_pred, region_pred, depth_pred)

# --------------------------------------------------------------------
# LOSS AND TRAINING CODE REMAINS THE SAME
# --------------------------------------------------------------------

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
    
    def log_model_weights_and_grads(self, epoch):
        """
        Log histograms of the parameters and their gradients.
        """
        for name, param in self.model.named_parameters():
            # Log parameter values
            self.writer.add_histogram(f'weights/{name}', param, epoch)
            # Log gradients (if they exist)
            if param.grad is not None:
                self.writer.add_histogram(f'grads/{name}', param.grad, epoch)
    
    def log_pr_curves(self, predictions, targets, epoch, prefix=''):
        """
        Log precision-recall curves for each task, if multi-class we do it for class 0 or 1, 
        you can adapt for each class as needed.
        """
        for task in ['condition', 'region', 'depth']:
            preds_softmax = F.softmax(predictions[task], dim=1)
            # Let's just pick the probability of 'class 1' for demonstration
            if preds_softmax.shape[1] > 1:
                positive_probs = preds_softmax[:, 1]
            else:
                positive_probs = preds_softmax[:, 0]
            
            self.writer.add_pr_curve(
                f'{prefix}PR/{task}', 
                labels=targets[task], 
                predictions=positive_probs, 
                global_step=epoch
            )
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        task_correct = {'condition': 0, 'region': 0, 'depth': 0}
        task_total = {'condition': 0, 'region': 0, 'depth': 0}
        task_losses = {'condition': 0, 'region': 0, 'depth': 0}
        
        for batch_idx, batch in enumerate(
            tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}", position=0, leave=True)
        ):
            images = batch['image'].to(self.device)
            targets = {
                task: batch[task].to(self.device)
                for task in ['condition', 'region', 'depth']
            }
            
            if epoch == 0 and batch_idx == 0:
                self.log_images(images, epoch)
            
            self.optimizer.zero_grad()
            
            condition_pred, region_pred, depth_pred = self.model(images)
            predictions = {
                'condition': condition_pred,
                'region': region_pred,
                'depth': depth_pred
            }
            
            loss, individual_losses = self.criterion(predictions, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            for task, t_loss in individual_losses.items():
                task_losses[task] += t_loss.item()
            
            # Calculate accuracy
            for task in ['condition', 'region', 'depth']:
                pred = torch.argmax(predictions[task], dim=1)
                task_correct[task] += (pred == targets[task]).sum().item()
                task_total[task] += targets[task].size(0)
            
            step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Batch/Loss', loss.item(), step)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracies = {
            task: task_correct[task] / task_total[task]
            for task in task_correct.keys()
        }
        avg_task_losses = {
            task: val / len(self.train_loader)
            for task, val in task_losses.items()
        }
        
        self.writer.add_scalar('Loss/train', avg_loss, epoch)
        for task in accuracies.keys():
            self.writer.add_scalar(f'Accuracy/train/{task}', accuracies[task], epoch)
            self.writer.add_scalar(f'Loss/train/{task}', avg_task_losses[task], epoch)
        
        return avg_loss, accuracies, predictions, targets
    
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        task_correct = {'condition': 0, 'region': 0, 'depth': 0}
        task_total = {'condition': 0, 'region': 0, 'depth': 0}
        task_losses = {'condition': 0, 'region': 0, 'depth': 0}
        
        # We'll store final predictions across the entire val set for PR curves
        final_preds = { 'condition': [], 'region': [], 'depth': [] }
        final_targets = { 'condition': [], 'region': [], 'depth': [] }
        
        for batch_idx, batch in enumerate(
            tqdm(self.val_loader, desc=f"Validation Epoch {epoch+1}", position=0, leave=True)
        ):
            images = batch['image'].to(self.device)
            targets = {
                task: batch[task].to(self.device)
                for task in ['condition', 'region', 'depth']
            }
            
            with torch.no_grad():
                condition_pred, region_pred, depth_pred = self.model(images)
            
            predictions = {
                'condition': condition_pred,
                'region': region_pred,
                'depth': depth_pred
            }
            
            loss, individual_losses = self.criterion(predictions, targets)
            total_loss += loss.item()
            
            for task, t_loss in individual_losses.items():
                task_losses[task] += t_loss.item()
            
            # Accuracy
            for task in ['condition', 'region', 'depth']:
                pred = torch.argmax(predictions[task], dim=1)
                task_correct[task] += (pred == targets[task]).sum().item()
                task_total[task] += targets[task].size(0)
            
            # Save preds/targets for entire epoch
            for task in ['condition', 'region', 'depth']:
                final_preds[task].append(predictions[task].cpu())
                final_targets[task].append(targets[task].cpu())
        
        # Convert final_preds, final_targets to single tensors
        for task in ['condition', 'region', 'depth']:
            final_preds[task] = torch.cat(final_preds[task], dim=0)
            final_targets[task] = torch.cat(final_targets[task], dim=0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracies = {
            task: task_correct[task] / task_total[task]
            for task in task_correct.keys()
        }
        avg_task_losses = {
            task: val / len(self.val_loader)
            for task, val in task_losses.items()
        }
        
        self.writer.add_scalar('Loss/val', avg_loss, epoch)
        for task in accuracies.keys():
            self.writer.add_scalar(f'Accuracy/val/{task}', accuracies[task], epoch)
            self.writer.add_scalar(f'Loss/val/{task}', avg_task_losses[task], epoch)
        
        return avg_loss, accuracies, final_preds, final_targets
    
    def train(self):
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Log model graph (try-catch if it doesn't support dict outputs)
        sample_input = next(iter(self.train_loader))['image'].to(self.device)
        try:
            self.writer.add_graph(self.model, sample_input, use_strict_trace=False)
        except Exception as e:
            print(f"Warning: Could not add model graph to TensorBoard: {str(e)}")
        
        for epoch in range(self.num_epochs):
            # Training
            train_loss, train_acc, train_preds, train_targets = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_acc, val_preds, val_targets = self.validate(epoch)
            
            # Log confusion matrices every 5 epochs
            if epoch % 5 == 0:
                self.log_confusion_matrices(epoch)
            
            self.log_model_weights_and_grads(epoch)
            self.log_pr_curves(val_preds, val_targets, epoch, prefix='val_')
            
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

def test_model(model, test_loader, device, 
               condition_num_classes=2, 
               region_num_classes=3, 
               depth_num_classes=2):
    """
    Evaluate a multi-task model on a test_loader and print extensive metrics.
    
    Args:
        model: PyTorch model returning (condition_pred, region_pred, depth_pred)
        test_loader: DataLoader for the test dataset
        device: 'cpu' or 'cuda'
        condition_num_classes: number of classes for condition
        region_num_classes: number of classes for region
        depth_num_classes: number of classes for depth
    """
    
    model.eval()
    
    from sklearn.metrics import accuracy_score
    
    # We'll store predictions and labels for each task
    all_condition_preds = []
    all_condition_labels = []
    
    all_region_preds = []
    all_region_labels = []
    
    all_depth_preds = []
    all_depth_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            condition_labels = batch['condition'].to(device)
            region_labels    = batch['region'].to(device)
            depth_labels     = batch['depth'].to(device)
            
            # Forward pass
            condition_pred, region_pred, depth_pred = model(images)
            
            # Take argmax across classes to get predicted labels
            condition_pred_labels = torch.argmax(condition_pred, dim=1)
            region_pred_labels    = torch.argmax(region_pred,    dim=1)
            depth_pred_labels     = torch.argmax(depth_pred,     dim=1)
            
            # Store predictions and ground truth on CPU
            all_condition_preds.extend(condition_pred_labels.cpu().numpy())
            all_condition_labels.extend(condition_labels.cpu().numpy())
            
            all_region_preds.extend(region_pred_labels.cpu().numpy())
            all_region_labels.extend(region_labels.cpu().numpy())
            
            all_depth_preds.extend(depth_pred_labels.cpu().numpy())
            all_depth_labels.extend(depth_labels.cpu().numpy())
    
    # Convert to NumPy arrays
    all_condition_preds = np.array(all_condition_preds)
    all_condition_labels = np.array(all_condition_labels)
    
    all_region_preds = np.array(all_region_preds)
    all_region_labels = np.array(all_region_labels)
    
    all_depth_preds = np.array(all_depth_preds)
    all_depth_labels = np.array(all_depth_labels)
    
    # Calculate accuracies
    condition_acc = accuracy_score(all_condition_labels, all_condition_preds)
    region_acc    = accuracy_score(all_region_labels,    all_region_preds)
    depth_acc     = accuracy_score(all_depth_labels,     all_depth_preds)
    
    # Confusion matrices
    cm_condition = confusion_matrix(all_condition_labels, all_condition_preds)
    cm_region    = confusion_matrix(all_region_labels,    all_region_preds)
    cm_depth     = confusion_matrix(all_depth_labels,     all_depth_preds)
    
    # Classification reports
    report_condition = classification_report(
        all_condition_labels, all_condition_preds, 
        labels=range(condition_num_classes),
        target_names=[f"class_{i}" for i in range(condition_num_classes)], 
        zero_division=0
    )
    report_region = classification_report(
        all_region_labels, all_region_preds,
        labels=range(region_num_classes),
        target_names=[f"class_{i}" for i in range(region_num_classes)],
        zero_division=0
    )
    report_depth = classification_report(
        all_depth_labels, all_depth_preds,
        labels=range(depth_num_classes),
        target_names=[f"class_{i}" for i in range(depth_num_classes)],
        zero_division=0
    )
    
    # Print results
    print("===== TEST RESULTS =====")
    
    print("Condition Results")
    print(f"  Accuracy: {condition_acc:.4f}")
    print("  Confusion Matrix:\n", cm_condition)
    print("  Classification Report:\n", report_condition)
    
    print("\nRegion Results")
    print(f"  Accuracy: {region_acc:.4f}")
    print("  Confusion Matrix:\n", cm_region)
    print("  Classification Report:\n", report_region)
    
    print("\nDepth Results")
    print(f"  Accuracy: {depth_acc:.4f}")
    print("  Confusion Matrix:\n", cm_depth)
    print("  Classification Report:\n", report_depth)
    
    return {
        'condition_acc': condition_acc,
        'region_acc': region_acc,
        'depth_acc': depth_acc,
        'cm_condition': cm_condition,
        'cm_region': cm_region,
        'cm_depth': cm_depth,
        'report_condition': report_condition,
        'report_region': report_region,
        'report_depth': report_depth
    }

def objective(trial):
    """
    Objective function for Optuna. It samples hyperparameters, trains the model, 
    and returns a validation metric (here we use validation loss) for Optuna to minimize.
    """
    # -----------------------------
    # 1) Suggest hyperparameters
    # -----------------------------
    learning_rate = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay  = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    batch_size    = trial.suggest_categorical('batch_size', [16, 32, 64])
    dropout_head  = trial.suggest_float('dropout_head', 0.3, 0.6)

    max_epochs    = 15
    patience      = 5

    # -----------------------------
    # 2) Set up data
    # -----------------------------
    base_dir = Path("/projectnb/nickar/kelvin/ASD_TolBlue")  # Adjust path to your data
    metadata_df = pd.read_csv('metadata/dataset_metadata.csv')
    
    # Simple train/val split
    metadata_df_shuffled = metadata_df.sample(frac=1, random_state=42).reset_index(drop=True)
    val_fraction = 0.2
    val_size = int(len(metadata_df_shuffled) * val_fraction)
    train_data = metadata_df_shuffled.iloc[val_size:].reset_index(drop=True)
    val_data   = metadata_df_shuffled.iloc[:val_size].reset_index(drop=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    axon_augmenter = AxonAugmenter(
        rotation_range=360,
        brightness_range=(0.8, 1.2),
        contrast_range=(0.8, 1.2),
        noise_range=(0.0, 0.05),
        blur_prob=0.3,
        blur_radius=(0.5, 1.5),
        zoom_range=(0.85, 1.15),
        flip_prob=0.5
    )

    train_dataset = AxonDataset(train_data, base_dir, transform=transform, augment=axon_augmenter)
    val_dataset   = AxonDataset(val_data,   base_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -----------------------------
    # 3) Build model & optimizer
    # -----------------------------
    model = MultiTaskEfficientNet().to(device)
    
    # Example: set new dropout with the trial-suggested dropout value
    # We'll simply locate the dropout layers in each head and adjust them
    for block in [model.condition_head, model.region_head, model.depth_head]:
        for i, layer in enumerate(block):
            if isinstance(layer, nn.Dropout):
                block[i] = nn.Dropout(p=dropout_head)
    
    criterion = MultiTaskLoss(weights={'condition': 1.0, 'region': 1.0, 'depth': 1.0})
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    exp_name = f"optuna_trial_{trial.number}"

    # -----------------------------
    # 4) Train 
    # -----------------------------
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=max_epochs,
        early_stopping_patience=patience,
        exp_name=exp_name
    )
    history = trainer.train()

    best_val_loss = min(history['val_loss'])
    return best_val_loss

def main():
    # Fix random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Create an Optuna study
    study = optuna.create_study(direction="minimize")  # We are minimizing val_loss
    
    # Run the study
    study.optimize(objective, n_trials=10, timeout=None, show_progress_bar=True)
    
    # Print summary of results
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Best trial:")
    best_trial = study.best_trial
    print(f"    Value (best val_loss): {best_trial.value}")
    print("    Params: ")
    for key, value in best_trial.params.items():
        print(f"      {key}: {value}")
    
    # -----------------------------
    # (Optional) Retrain or load the best model
    # -----------------------------
    best_hparams = best_trial.params
    print("\nRe-building the best model to test on hold-out set...")

    base_dir = Path("/projectnb/nickar/kelvin/ASD_TolBlue")
    metadata_df = pd.read_csv('metadata/dataset_metadata.csv')
    # Suppose your CSV indicates a test split with "split='test'"
    test_data = metadata_df[metadata_df['split'] == 'test'].reset_index(drop=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    test_dataset = AxonDataset(test_data, base_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=int(best_hparams['batch_size']), shuffle=False)

    model = MultiTaskEfficientNet().to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for testing.")
        model = nn.DataParallel(model)

    # Adjust the dropout layers based on best hyperparameters
    dropout_head = best_hparams['dropout_head']
    for block in [model.module.condition_head if isinstance(model, nn.DataParallel) else model.condition_head,
                  model.module.region_head    if isinstance(model, nn.DataParallel) else model.region_head,
                  model.module.depth_head     if isinstance(model, nn.DataParallel) else model.depth_head]:
        for i, layer in enumerate(block):
            if isinstance(layer, nn.Dropout):
                block[i] = nn.Dropout(p=dropout_head)

    # Re-create optimizer, etc., if you plan to retrain. Or just load from 'best_model.pth':
    # checkpoint = torch.load('best_model.pth')
    # model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate on the test set:
    metrics = test_model(
        model=model,
        test_loader=test_loader,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        condition_num_classes=2,  # CTR vs ASD
        region_num_classes=3,     # A25, A46, OFC
        depth_num_classes=2       # DWM vs SWM
    )
    print("Test Metrics (with best hyperparams):", metrics)

if __name__ == "__main__":
    main()