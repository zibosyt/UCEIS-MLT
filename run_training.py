#!/usr/bin/env python3
"""
Five-fold cross-validation training script
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import timm
import os
import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import sys

from SimAM import ChannelAttention, insert_attention_to_coatnet
from multi_task_model import ImprovedMultiTaskModel, create_model
from model_config import get_backbone_config, get_optimal_config_for_gpu


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance, supporting label smoothing and class weights"""
    def __init__(self, alpha=None, gamma=2, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.ce = nn.CrossEntropyLoss(
            reduction='none', 
            label_smoothing=label_smoothing,
            weight=alpha
        )

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_class_weights(class_counts):
    """Calculate class weights to handle class imbalance"""
    total = sum(class_counts)
    weights = [total / c for c in class_counts]
    return torch.tensor(weights, dtype=torch.float32)


def compute_class_weights_from_dataset(dataset, tasks=['vascular', 'bleeding', 'ulceration']):
    """Dynamically calculate class weights for all tasks from dataset"""
    weights_dict = {}
    
    for task in tasks:
        # Count samples for each class
        class_counts = dataset.data_frame[task].value_counts().sort_index()
        # Ensure all classes are included (even with zero count)
        max_class = max(class_counts.index)
        counts_list = [class_counts.get(i, 0) for i in range(max_class + 1)]
        # Calculate weights
        weights = get_class_weights(counts_list)
        weights_dict[task] = weights
        
        # Print weight information
        print(f"{task} class distribution: {counts_list}")
        print(f"{task} class weights: {weights}")
    
    return weights_dict


class DynamicWeightedLoss(nn.Module):
    """Dynamically adjust task weights"""
    def __init__(self, num_tasks):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, losses):
        weights = torch.exp(-self.log_vars)
        weighted_losses = weights * losses + self.log_vars
        return torch.sum(weighted_losses)


def find_image_file(images_dir, img_name):
    """Find image file, handling case sensitivity issues"""
    # First try direct path
    img_path = os.path.join(images_dir, img_name)
    if os.path.exists(img_path):
        return img_path
    
    # If direct path doesn't exist, try different extension variations
    base_name, ext = os.path.splitext(img_name)
    extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']  # Common extensions
    
    for extension in extensions:
        potential_path = os.path.join(images_dir, base_name + extension)
        if os.path.exists(potential_path):
            return potential_path
    
    # If none found, return original path (let subsequent processing error)
    return os.path.join(images_dir, img_name)


class UCEISMultiTaskDataset(Dataset):
    """UCEIS multi-task dataset class"""
    def __init__(self, csv_file, images_dir='./images', transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data_frame.iloc[idx, 2]  # image_path column
        # Use helper function to find image file
        img_path = find_image_file(self.images_dir, str(img_name))
        
        # Read image
        image = Image.open(img_path).convert('RGB')
        
        # Get labels for three tasks
        vascular = self.data_frame.iloc[idx, 5]  # vascular column
        bleeding = self.data_frame.iloc[idx, 6]  # bleeding column
        ulceration = self.data_frame.iloc[idx, 7]  # ulceration column
        
        # Validate and clamp label values within reasonable ranges
        # vascular: 0-2
        vascular = max(0, min(2, int(vascular)))
        # bleeding: 0-3
        bleeding = max(0, min(3, int(bleeding)))
        # ulceration: 0-3
        ulceration = max(0, min(3, int(ulceration)))

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(vascular), torch.tensor(bleeding), torch.tensor(ulceration)


def calculate_accuracy(predictions, targets):
    """计算准确率"""
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    return correct / total


def train_coatnet3_fold(fold, backbone_name='coatnet_3_rw_224', use_simam=False):
    """Train single fold using CoAtNet-3
    
    Args:
        fold: Fold number (0-4)
        backbone_name: Backbone network name
        use_simam: Whether to use SimAM attention mechanism
                  - None: Auto-determine (based on GPU memory)
                  - True: Force use
                  - False: Force not use
    """
    print(f"===== Fold {fold+1} CoAtNet-3 Training Started =====")
    print(f"Using backbone network: {backbone_name}")
    
    # Get backbone network configuration
    backbone_config = get_backbone_config(backbone_name)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU memory: {gpu_memory:.1f}GB")
        
        # Auto-determine whether to use SimAM
        if use_simam is None:
            if gpu_memory >= 16:
                use_simam = True
                print("GPU memory sufficient, enabling SimAM attention mechanism")
            elif gpu_memory >= 12:
                use_simam = True  # Try first, will disable if OOM
                print("GPU memory moderate, attempting to enable SimAM (will auto retry if out of memory)")
            else:
                use_simam = False
                print("GPU memory limited, disabling SimAM to save memory")
    else:
        use_simam = False if use_simam is None else use_simam
    
    print(f"SimAM attention mechanism: {'Enabled' if use_simam else 'Disabled'}")
    
    # File paths - differentiate filenames based on SimAM usage
    train_file = f'kfold/train_fold{fold}.csv'
    val_file = f'kfold/val_fold{fold}.csv'
    simam_suffix = '_simam' if use_simam else '_no_simam'
    saved_model_file = f'models/best_coatnet3_fold{fold}{simam_suffix}.pth'
    
    # Check if data files exist
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not os.path.exists(val_file):
        raise FileNotFoundError(f"Validation file not found: {val_file}")
    
    # Ensure model save directory exists
    os.makedirs('models', exist_ok=True)
    
    # Adjust batch size based on configuration
    if backbone_config:
        batch_size = backbone_config['batch_size']
        base_lr = backbone_config['lr']
        print(f"Using recommended configuration - Batch size: {batch_size}, Learning rate: {base_lr}")
    else:
        batch_size = 24  # CoAtNet-3 default batch size
        base_lr = 8e-5   # CoAtNet-3 default learning rate
        print(f"Using default configuration - Batch size: {batch_size}, Learning rate: {base_lr}")
    
    # Data augmentation and preprocessing - optimized for CoAtNet-3
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.15),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = UCEISMultiTaskDataset(
        csv_file=train_file,
        images_dir='./images',
        transform=train_transform
    )
    
    val_dataset = UCEISMultiTaskDataset(
        csv_file=val_file,
        images_dir='./images',
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    
    # Create CoAtNet-3 model - Pure CoAtNet recommended
    print("Creating CoAtNet-3 model...")
    
    # Prefer pure CoAtNet (no additional attention)
    use_pure_coatnet = True  # Set to True to use pure CoAtNet
    
    try:
        if use_pure_coatnet:
            from coatnet_training import create_pure_coatnet_model
            model = create_pure_coatnet_model(
                backbone_name=backbone_name,
                num_classes_per_task=(3, 4, 4),
                pretrained=True,
                dropout_rate=0.5
            )
            print("Using pure CoAtNet model (recommended)")
        else:
            model = create_model(
                backbone_name=backbone_name,
                num_classes_per_task=(3, 4, 4),
                pretrained=True,
                dropout_rate=0.5,
                use_attention=use_simam
            )
            print(f"Using model with additional attention (SimAM: {use_simam})")
        
        model = model.to(device)
    
        # Print model information
        model_info = model.get_model_info()
        print(f"Model information:")
        print(f"  - Backbone: {model_info['backbone']}")
        print(f"  - Total parameters: {model_info['total_params']:,}")
        print(f"  - Trainable parameters: {model_info['trainable_params']:,}")
        print(f"  - Feature dimension: {model_info['feature_dim']}")
        print(f"  - SimAM attention: {'Enabled' if use_simam else 'Disabled'}")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and use_simam:
            print(f"GPU out of memory, attempting to create model without SimAM...")
            torch.cuda.empty_cache()  # Clear GPU cache
            
            model = create_model(
                backbone_name=backbone_name,
                num_classes_per_task=(3, 4, 4),
                pretrained=True,
                dropout_rate=0.5,
                use_attention=False  # Disable SimAM
            )
            model = model.to(device)
            
            # Update save filename
            saved_model_file = f'models/best_coatnet3_fold{fold}_no_simam.pth'
            use_simam = False
            
            print(f"Successfully created model without SimAM")
        else:
            raise e
    
    # Calculate class weights
    print("Calculating class weights...")
    class_weights = compute_class_weights_from_dataset(train_dataset)
    
    # Move weights to device
    vascular_weights = class_weights['vascular'].to(device)
    bleeding_weights = class_weights['bleeding'].to(device)
    ulceration_weights = class_weights['ulceration'].to(device)
    
    # Define loss functions (using Focal Loss)
    criterion_vascular = FocalLoss(alpha=vascular_weights, gamma=2, label_smoothing=0.1)
    criterion_bleeding = FocalLoss(alpha=bleeding_weights, gamma=2, label_smoothing=0.1)
    criterion_ulceration = FocalLoss(alpha=ulceration_weights, gamma=2, label_smoothing=0.1)
    
    # Define dynamic weighted loss
    dynamic_loss = DynamicWeightedLoss(num_tasks=3)
    dynamic_loss = dynamic_loss.to(device)
    
    # Training configuration
    num_epochs = 100
    warmup_epochs = 3
    min_lr = base_lr / 100
    weight_decay = 1e-4
    max_grad_norm = 1.0
    
    # Define optimizer (AdamW)
    optimizer = optim.AdamW([
        {'params': model.parameters(), 'lr': base_lr},
        {'params': dynamic_loss.parameters(), 'lr': base_lr}
    ], weight_decay=weight_decay)
    
    # Learning rate scheduler (Cosine Annealing with Warmup)
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs - warmup_epochs, 
        eta_min=min_lr
    )
    
    # Training state tracking
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    start_epoch = 0
    
    # Check for saved training state
    checkpoint_file = f'models/checkpoint_coatnet3_fold{fold}.pth'
    if os.path.exists(checkpoint_file):
        print(f"Found saved training state, attempting to load...")
        try:
            checkpoint = torch.load(checkpoint_file)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            patience_counter = checkpoint['patience_counter']
            print(f"Successfully loaded training state, will start from epoch {start_epoch}")
            print(f"   Previous best validation loss: {best_val_loss:.4f}")
            print(f"   Previous patience counter: {patience_counter}/{patience}")
        except Exception as e:
            print(f"Failed to load training state: {e}")
            print("   Will start training from scratch")
    
    # Learning rate warmup
    def warmup_lr(optimizer, epoch, warmup_epochs, base_lr):
        """Linear learning rate warmup"""
        if epoch < warmup_epochs:
            lr = base_lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            scheduler.step()
    
    print("Starting training...")
    
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        running_vascular_loss = 0.0
        running_bleeding_loss = 0.0
        running_ulceration_loss = 0.0
        running_total_loss = 0.0
        
        vascular_train_acc = 0.0
        bleeding_train_acc = 0.0
        ulceration_train_acc = 0.0
        
        for batch_idx, (images, vascular_labels, bleeding_labels, ulceration_labels) in enumerate(train_loader):
            images = images.to(device)
            vascular_labels = vascular_labels.to(device)
            bleeding_labels = bleeding_labels.to(device)
            ulceration_labels = ulceration_labels.to(device)
            
            # Check label ranges
            if vascular_labels.max() >= 3 or vascular_labels.min() < 0:
                print(f"WARNING: vascular label values out of range: {vascular_labels}")
            if bleeding_labels.max() >= 4 or bleeding_labels.min() < 0:
                print(f"WARNING: bleeding label values out of range: {bleeding_labels}")
            if ulceration_labels.max() >= 4 or ulceration_labels.min() < 0:
                print(f"WARNING: ulceration label values out of range: {ulceration_labels}")
            
            optimizer.zero_grad()
            
            # Forward pass
            vascular_out, bleeding_out, ulceration_out = model(images)
            
            # Check for nan values in outputs
            if torch.isnan(vascular_out).any() or torch.isnan(bleeding_out).any() or torch.isnan(ulceration_out).any():
                print(f"Batch {batch_idx} detected nan values, skipping")
                continue
            
            # Calculate losses for each task
            try:
                vascular_loss = criterion_vascular(vascular_out, vascular_labels)
                bleeding_loss = criterion_bleeding(bleeding_out, bleeding_labels)
                ulceration_loss = criterion_ulceration(ulceration_out, ulceration_labels)
            except RuntimeError as e:
                print(f"Batch {batch_idx} loss calculation error: {e}")
                continue
            
            # Check for nan values in losses
            if torch.isnan(vascular_loss) or torch.isnan(bleeding_loss) or torch.isnan(ulceration_loss):
                print(f"Batch {batch_idx} detected nan losses, skipping")
                continue
            
            # Use dynamic weighted loss
            total_loss = dynamic_loss(torch.stack([vascular_loss, bleeding_loss, ulceration_loss]))
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            
            # Accumulate losses
            running_vascular_loss += vascular_loss.item()
            running_bleeding_loss += bleeding_loss.item()
            running_ulceration_loss += ulceration_loss.item()
            running_total_loss += total_loss.item()
            
            # Calculate accuracies
            vascular_train_acc += calculate_accuracy(vascular_out, vascular_labels)
            bleeding_train_acc += calculate_accuracy(bleeding_out, bleeding_labels)
            ulceration_train_acc += calculate_accuracy(ulceration_out, ulceration_labels)
        
        # Calculate average training metrics
        if len(train_loader) > 0:
            avg_vascular_train_loss = running_vascular_loss / len(train_loader)
            avg_bleeding_train_loss = running_bleeding_loss / len(train_loader)
            avg_ulceration_train_loss = running_ulceration_loss / len(train_loader)
            avg_total_train_loss = running_total_loss / len(train_loader)
            
            avg_vascular_train_acc = vascular_train_acc / len(train_loader)
            avg_bleeding_train_acc = bleeding_train_acc / len(train_loader)
            avg_ulceration_train_acc = ulceration_train_acc / len(train_loader)
        else:
            avg_vascular_train_loss = avg_bleeding_train_loss = avg_ulceration_train_loss = avg_total_train_loss = float('nan')
            avg_vascular_train_acc = avg_bleeding_train_acc = avg_ulceration_train_acc = 0.0
        
        # Validation phase
        model.eval()
        val_vascular_loss = 0.0
        val_bleeding_loss = 0.0
        val_ulceration_loss = 0.0
        val_total_loss = 0.0
        
        val_vascular_acc = 0.0
        val_bleeding_acc = 0.0
        val_ulceration_acc = 0.0
        
        with torch.no_grad():
            for images, vascular_labels, bleeding_labels, ulceration_labels in val_loader:
                images = images.to(device)
                vascular_labels = vascular_labels.to(device)
                bleeding_labels = bleeding_labels.to(device)
                ulceration_labels = ulceration_labels.to(device)
                
                # Forward pass
                vascular_out, bleeding_out, ulceration_out = model(images)
                
                # Check for nan values in outputs
                if torch.isnan(vascular_out).any() or torch.isnan(bleeding_out).any() or torch.isnan(ulceration_out).any():
                    continue
                
                # Calculate losses
                vascular_loss = criterion_vascular(vascular_out, vascular_labels)
                bleeding_loss = criterion_bleeding(bleeding_out, bleeding_labels)
                ulceration_loss = criterion_ulceration(ulceration_out, ulceration_labels)
                
                # Check for nan values in losses
                if torch.isnan(vascular_loss) or torch.isnan(bleeding_loss) or torch.isnan(ulceration_loss):
                    continue
                
                # Accumulate losses
                val_vascular_loss += vascular_loss.item()
                val_bleeding_loss += bleeding_loss.item()
                val_ulceration_loss += ulceration_loss.item()
                val_total_loss += (vascular_loss + bleeding_loss + ulceration_loss).item()
                
                # Calculate accuracies
                val_vascular_acc += calculate_accuracy(vascular_out, vascular_labels)
                val_bleeding_acc += calculate_accuracy(bleeding_out, bleeding_labels)
                val_ulceration_acc += calculate_accuracy(ulceration_out, ulceration_labels)
        
        # Calculate average validation metrics
        if len(val_loader) > 0:
            avg_val_vascular_loss = val_vascular_loss / len(val_loader)
            avg_val_bleeding_loss = val_bleeding_loss / len(val_loader)
            avg_val_ulceration_loss = val_ulceration_loss / len(val_loader)
            avg_val_total_loss = val_total_loss / len(val_loader)
            
            avg_val_vascular_acc = val_vascular_acc / len(val_loader)
            avg_val_bleeding_acc = val_bleeding_acc / len(val_loader)
            avg_val_ulceration_acc = val_ulceration_acc / len(val_loader)
        else:
            avg_val_vascular_loss = avg_val_bleeding_loss = avg_val_ulceration_loss = avg_val_total_loss = float('nan')
            avg_val_vascular_acc = avg_val_bleeding_acc = avg_val_ulceration_acc = 0.0
        
        # Update learning rate
        warmup_lr(optimizer, epoch, warmup_epochs, base_lr)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print training information
        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time, '%Y-%m-%d %H:%M:%S')
        
        print(f'{time_str} - Epoch [{epoch+1}/{num_epochs}] - LR: {current_lr:.6f}')
        print(f'Train - Total Loss: {avg_total_train_loss:.4f}, '
              f'Vascular Loss: {avg_vascular_train_loss:.4f}, '
              f'Bleeding Loss: {avg_bleeding_train_loss:.4f}, '
              f'Ulceration Loss: {avg_ulceration_train_loss:.4f}')
        print(f'Train - Vascular Acc: {avg_vascular_train_acc:.4f}, '
              f'Bleeding Acc: {avg_bleeding_train_acc:.4f}, '
              f'Ulceration Acc: {avg_ulceration_train_acc:.4f}')
        print(f'Val - Total Loss: {avg_val_total_loss:.4f}, '
              f'Vascular Loss: {avg_val_vascular_loss:.4f}, '
              f'Bleeding Loss: {avg_val_bleeding_loss:.4f}, '
              f'Ulceration Loss: {avg_val_ulceration_loss:.4f}')
        print(f'Val - Vascular Acc: {avg_val_vascular_acc:.4f}, '
              f'Bleeding Acc: {avg_val_bleeding_acc:.4f}, '
              f'Ulceration Acc: {avg_val_ulceration_acc:.4f}')
        print('-' * 100)
        
        # Save best model
        if avg_val_total_loss < best_val_loss and not torch.isnan(torch.tensor(avg_val_total_loss)):
            best_val_loss = avg_val_total_loss
            torch.save(model.state_dict(), saved_model_file)
            print(f'Saving best model to: {saved_model_file}')
            print(f'   Validation loss: {best_val_loss:.4f}')
            patience_counter = 0
        else:
            patience_counter += 1
            print(f'Validation loss not improved, patience counter: {patience_counter}/{patience}')
        
        # Save training state (at the end of each epoch)
        checkpoint_file = f'models/checkpoint_coatnet3_fold{fold}.pth'
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'base_lr': base_lr,
            'num_epochs': num_epochs,
            'warmup_epochs': warmup_epochs,
            'min_lr': min_lr
        }
        torch.save(checkpoint, checkpoint_file)
        
        # Early stopping
        if patience_counter >= patience:
            print(f'Reached patience limit, stopping training early')
            break
    
    # Ensure at least the last epoch model is saved
    if not os.path.exists(saved_model_file):
        torch.save(model.state_dict(), saved_model_file)
        print(f'Saving last epoch model to: {saved_model_file}')
    
    print(f"Fold {fold+1} training completed! Best validation loss: {best_val_loss:.4f}")
    return best_val_loss


def check_dependencies():
    """Check dependencies and data files"""
    print("Checking dependencies and data files...")
    
    # Check required Python files
    required_files = [
        'multi_task_model.py',
        'model_config.py', 
        'SimAM.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"Missing files: {', '.join(missing_files)}")
        return False
    
    # Check kfold data folder
    if not os.path.exists('kfold'):
        print("Missing kfold folder, please ensure cross-validation data is prepared")
        print("   Required files: train_fold0.csv, val_fold0.csv, ... train_fold4.csv, val_fold4.csv")
        return False
    
    # Check specific fold files
    missing_fold_files = []
    for i in range(5):
        train_file = f'kfold/train_fold{i}.csv'
        val_file = f'kfold/val_fold{i}.csv'
        if not os.path.exists(train_file):
            missing_fold_files.append(train_file)
        if not os.path.exists(val_file):
            missing_fold_files.append(val_file)
    
    if missing_fold_files:
        print(f"Missing fold files: {missing_fold_files[:3]}{'...' if len(missing_fold_files) > 3 else ''}")
        return False
    
    # Check images directory
    images_dir = './images'
    if not os.path.exists(images_dir):
        print(f"Warning: Images directory does not exist: {images_dir}")
        print("   Please confirm the image path is correct")
        # Don't return False, allow user to modify path
    
    # Check model save folder
    if not os.path.exists('models'):
        print("Creating models folder...")
        os.makedirs('models')
    
    print("Dependencies check completed")
    return True


def get_gpu_info():
    """Get GPU information"""
    if not torch.cuda.is_available():
        return None, 0
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    return gpu_name, gpu_memory


def run_coatnet3_5fold_training(selected_folds=None):
    """Run CoAtNet-3 5-fold cross-validation training"""
    print("CoAtNet-3 5-fold Cross-Validation Training")
    print("=" * 60)
    
    # Get GPU information
    gpu_name, gpu_memory = get_gpu_info()
    
    if gpu_name:
        print(f"GPU: {gpu_name}")
        print(f"GPU memory: {gpu_memory:.1f}GB")
        
        # Check if GPU memory is sufficient
        if gpu_memory < 12:
            print("Warning: GPU memory may be insufficient, CoAtNet-3 requires large memory")
            print("   Recommended GPU with 12GB+ memory")
            
            response = input("Continue training? (y/N): ").strip().lower()
            if response != 'y':
                print("Training cancelled")
                return
    else:
        print("No GPU detected, CoAtNet-3 requires GPU for training")
        return
    
    # Training configuration information
    backbone_config = get_backbone_config('coatnet_3_rw_224')
    if backbone_config:
        print(f"\nCoAtNet-3 Configuration:")
        print(f"   Description: {backbone_config['description']}")
        print(f"   Parameters: {backbone_config['params']}")
        print(f"   Recommended batch size: {backbone_config['batch_size']}")
        print(f"   Recommended learning rate: {backbone_config['lr']}")
        print(f"   Memory usage: {backbone_config['memory_usage']}")
        print(f"   Training time: {backbone_config['training_time']}")
    
    # Select whether to use SimAM attention mechanism
    use_simam = False
   
    # Confirm start training
    
    print(f"\n" + "="*60)
    print(f"Starting training...")
    print(f"="*60)
    
    # Record training results
    fold_results = []
    successful_folds = []
    failed_folds = []
    
    start_time = datetime.datetime.now()
    
    # Determine which folds to train
    if selected_folds is None:
        selected_folds = list(range(5))
    
    print(f"Training folds: {[fold+1 for fold in selected_folds]}")
    print(f"Using SimAM: {'Auto' if use_simam is None else 'Yes' if use_simam else 'No'}")
    
    for i, fold in enumerate(selected_folds):
        try:
            print(f"\n{'='*20} Fold {fold+1} / {len(selected_folds)} {'='*20}")
            
            fold_start_time = datetime.datetime.now()
            best_val_loss = train_coatnet3_fold(fold, 'coatnet_3_rw_224', use_simam)
            fold_end_time = datetime.datetime.now()
            fold_duration = fold_end_time - fold_start_time
            
            fold_results.append({
                'fold': fold + 1,
                'best_val_loss': best_val_loss,
                'duration': fold_duration,
                'status': 'success'
            })
            successful_folds.append(fold + 1)
            
            print(f"Fold {fold+1} training completed")
            print(f"   Best validation loss: {best_val_loss:.4f}")
            print(f"   Training duration: {fold_duration}")
            
        except Exception as e:
            print(f"Fold {fold+1} training failed: {e}")
            fold_results.append({
                'fold': fold + 1,
                'best_val_loss': float('inf'),
                'duration': datetime.timedelta(0),
                'status': 'failed',
                'error': str(e)
            })
            failed_folds.append(fold + 1)
            
            # Automatically continue training remaining folds
            if i < len(selected_folds) - 1:  # Not the last fold
                print(f"Fold {fold+1} training failed, continuing with remaining folds")
    
    end_time = datetime.datetime.now()
    total_duration = end_time - start_time
    
    # Training summary
    print(f"\n" + "="*60)
    print(f"CoAtNet-3 5-fold Cross-Validation Training Summary")
    print(f"="*60)
    
    print(f"Total training duration: {total_duration}")
    print(f"Successful folds: {len(successful_folds)}/5")
    print(f"Failed folds: {len(failed_folds)}/5")
    
    if successful_folds:
        print(f"\nSuccessful folds:")
        valid_losses = []
        for result in fold_results:
            if result['status'] == 'success':
                print(f"   Fold {result['fold']}: Validation loss {result['best_val_loss']:.4f}, "
                      f"duration {result['duration']}")
                valid_losses.append(result['best_val_loss'])
        
        if valid_losses:
            avg_val_loss = np.mean(valid_losses)
            std_val_loss = np.std(valid_losses)
            print(f"\nPerformance statistics:")
            print(f"   Average validation loss: {avg_val_loss:.4f} ± {std_val_loss:.4f}")
            print(f"   Best validation loss: {min(valid_losses):.4f}")
            print(f"   Worst validation loss: {max(valid_losses):.4f}")
    
    if failed_folds:
        print(f"\nFailed folds: {failed_folds}")
        for result in fold_results:
            if result['status'] == 'failed':
                print(f"   Fold {result['fold']}: {result.get('error', 'Unknown error')}")
    
    print(f"\nModel files saved in models/ directory:")
    for fold in successful_folds:
        # Try two possible filename formats (with and without simam suffix)
        for suffix in ['_no_simam', '_simam', '']:
            model_file = f"models/best_coatnet3_fold{fold-1}{suffix}.pth"
            if os.path.exists(model_file):
                file_size = os.path.getsize(model_file) / (1024 * 1024)
                print(f"   {model_file} ({file_size:.1f} MB)")
                break
    
    if successful_folds:
        print(f"\nTraining completed! You can use evaluate_model5Fold.py for evaluation")
    else:
        print(f"\nAll folds failed to train, please check data and configuration")


def main():
    """Main function"""
    print("CoAtNet-3 UCEIS Multi-task Classification Training System")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("Dependency check failed, please resolve and try again")
        sys.exit(1)
    
    try:
        # Display main menu
        while True:
            print("\n" + "="*25 + " Main Menu " + "="*25)
            print("1. Start full 5-fold cross-validation training")
            print("2. Select specific folds for training")
            print("3. Exit system")
            print("="*60)
            
            choice = input("Please select an operation (1-3): ").strip()
            
            if choice == "1":
                run_coatnet3_5fold_training()
                break
            elif choice == "2":
                selected_folds = []
                while True:
                    fold_input = input("Please enter folds to train (1-5, comma-separated, or enter 'all' to select all): ").strip()
                    if fold_input.lower() == 'all':
                        selected_folds = list(range(5))
                        break
                    else:
                        try:
                            selected_folds = [int(fold.strip()) - 1 for fold in fold_input.split(',') if fold.strip()]
                            if all(0 <= fold < 5 for fold in selected_folds):
                                break
                            else:
                                print("Folds must be between 1-5")
                        except ValueError:
                            print("Please enter valid fold numbers")
                
                run_coatnet3_5fold_training(selected_folds)
                break
            elif choice == "3":
                print("Thank you for using, goodbye!")
                sys.exit(0)
            else:
                print("Invalid choice, please re-enter")
        
    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()