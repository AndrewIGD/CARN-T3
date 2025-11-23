import json
import torch
from torch import nn
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import numpy as np
from datetime import datetime
import argparse

from datasets import load_dataset
from models import load_model
from optimizers import SAM, Muon, get_optimizer
from lr_schedulers import get_lr_scheduler
from bs_schedulers import get_batch_size_scheduler
from transforms import get_transforms, get_cutmix_mixup
import random
import wandb

import torchvision.transforms.v2 as v2

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif (self.mode == 'min' and score < self.best_score - self.min_delta) or \
             (self.mode == 'max' and score > self.best_score + self.min_delta):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def train_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp, train_transforms, cutmix_mixup, normalize, epoch=None, lr=None, batch_size=None, prev_val_acc=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    desc_parts = ['Training']
    if epoch is not None:
        desc_parts.append(f'Epoch: {epoch}')
    if lr is not None:
        desc_parts.append(f'LR: {lr:.2e}')
    if batch_size is not None:
        desc_parts.append(f'BS: {batch_size}')
    if prev_val_acc is not None:
        desc_parts.append(f'Val Acc: {prev_val_acc:.2f}%')
    desc = ' | '.join(desc_parts)
    
    for inputs, targets in tqdm(train_loader, desc=desc, leave=False):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        inputs = train_transforms(inputs)
        
        if cutmix_mixup is not None:
            cutmix_transform, cutmix_p = cutmix_mixup
            if random.random() < cutmix_p:
                inputs, targets = cutmix_transform(inputs, targets)
        
        inputs = normalize(inputs)
        
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        if isinstance(optimizer, Muon):
            scaler.step(optimizer.base_optimizer)
            optimizer.base_optimizer.zero_grad()

            scaler.step(optimizer)
            optimizer.zero_grad()
        elif isinstance(optimizer, SAM):
            # Pretty confident that using amp enabled here is not a good idea
            optimizer.first_step(zero_grad=True)
            
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                second_outputs = model(inputs)
                second_loss = criterion(second_outputs, targets)
            
            scaler.scale(second_loss).backward()
            optimizer.second_step(zero_grad=True)
        else:
            scaler.step(optimizer)
            optimizer.zero_grad()
        scaler.update()
        
        running_loss += loss.item()
        predicted = outputs.argmax(1)
        total += targets.size(0)
        
        if targets.dim() > 1:
            target_classes = targets.argmax(1)
        else:
            target_classes = targets
        correct += predicted.eq(target_classes).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, val_loader, criterion, device, use_amp, tta_transforms_list, normalize, epoch=None, lr=None, batch_size=None, prev_val_acc=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    desc_parts = ['Validating']
    if epoch is not None:
        desc_parts.append(f'Epoch: {epoch}')
    if lr is not None:
        desc_parts.append(f'LR: {lr:.2e}')
    if batch_size is not None:
        desc_parts.append(f'BS: {batch_size}')
    if prev_val_acc is not None:
        desc_parts.append(f'Val Acc: {prev_val_acc:.2f}%')
    desc = ' | '.join(desc_parts)
    
    for inputs, targets in tqdm(val_loader, desc=desc, leave=False):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        batch_predictions = []
        
        if tta_transforms_list and len(tta_transforms_list) > 0:
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(tta_transforms_list[0](inputs.clone()))
                batch_predictions.append(outputs)
            
            for transform in tta_transforms_list:
                augmented_inputs = transform(inputs.clone())
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(augmented_inputs)
                batch_predictions.append(outputs)
        else:
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(normalize(inputs))
                batch_predictions.append(outputs)
        
        batch_predictions = torch.stack(batch_predictions)
        mean_predictions = batch_predictions.mean(dim=0)
        
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            loss = criterion(batch_predictions[0], targets)
        
        running_loss += loss.item()
        _, predicted = mean_predictions.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

def load_config(config_path='config'):
    with open(f'{config_path}.json', 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description='Train a model with specified configuration')
    parser.add_argument('-c', '--config', type=str, default='config',
                        help='Path to the configuration file (default: config)')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    dataset = config['dataset']
    model_name = config['model']
    pretrained = config.get('pretrained', False)
    epochs = config['epochs']
    batch_size = config['batch_size']
    max_batch_size = config.get('max_batch_size', 256)
    min_batch_size = config.get('min_batch_size', 1)
    lr = config['lr']
    optimizer_name = config['optimizer']
    scheduler_name = config.get('scheduler', 'cosine')
    batch_schedule = config.get('batch_schedule', 'none')
    early_stopping_enabled = config.get('early_stopping', False)
    use_amp = config.get('use_amp', False) and optimizer_name != 'sam'
    label_smoothing = config.get('label_smoothing', 0.0)
    
    optimizer_configs = config.get('optimizer_configs', {})
    scheduler_configs = config.get('scheduler_configs', {})
    batch_scheduler_config = config.get('batch_scheduler_config', {})
    early_stopping_config = config.get('early_stopping_config', {})
    
    device = get_device()
    
    use_amp = use_amp and device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    
    train_loader, val_loader, num_classes, mean, std = load_dataset(
        dataset,
        batch_size=batch_size
    )
    
    train_transforms = get_transforms(config, dataset, mean, std, train=True, model_name=model_name)
    tta_transforms_list = get_transforms(config, dataset, mean, std, train=False, model_name=model_name)

    normalize = v2.Normalize(mean, std, inplace=True)
    
    model = load_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    model = model.to(device)

    opt_config = optimizer_configs.get(optimizer_name, {})
    optimizer_kwargs = {'lr': lr}
    
    if 'betas' in opt_config and isinstance(opt_config['betas'], list):
        opt_config['betas'] = tuple(opt_config['betas'])
    
    optimizer_kwargs.update(opt_config)
    
    optimizer = get_optimizer(
        optimizer_name,
        model.parameters(),
        **optimizer_kwargs
    )
    
    scheduler = None
    sched_config = {}
    if scheduler_name:
        sched_config = scheduler_configs.get(scheduler_name, {})
        scheduler = get_lr_scheduler(
            scheduler_name,
            optimizer,
            **sched_config
        )
    
    batch_scheduler = None
    batch_sched_config = {}
    if batch_schedule != 'none':
        batch_sched_config = batch_scheduler_config.get(batch_schedule, {}).copy()
        batch_sched_config['max_batch_size'] = max_batch_size
        batch_sched_config['min_batch_size'] = min_batch_size
        batch_scheduler = get_batch_size_scheduler(
            batch_schedule,
            train_loader,
            **batch_sched_config
        )
    
    early_stopping = None
    if early_stopping_enabled:
        early_stopping = EarlyStopping(mode='max', **early_stopping_config)
    
    transforms_config = config.get('transforms', {})
    train_config = transforms_config.get('train', {})
    test_config = transforms_config.get('test', {})
    
    print("=" * 80)
    print(f"Training Configuration:")
    print(f"  Dataset: {dataset}")
    print(f"  Model: {model_name}")
    print(f"  Device: {device}")
    print(f"  Pretrained: {pretrained}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {lr}")
    print(f"  Optimizer: {optimizer_name}")
    for key, value in opt_config.items():
        print(f"    {key}: {value}")
    if scheduler_name:
        print(f"  LR Scheduler: {scheduler_name}")
        for key, value in sched_config.items():
            print(f"    {key}: {value}")
    if batch_schedule != 'none':
        print(f"  Batch Scheduler: {batch_schedule}")
        for key, value in batch_sched_config.items():
            print(f"    {key}: {value}")
    if early_stopping_enabled:
        print(f"  Early Stopping: enabled")
        for key, value in early_stopping_config.items():
            print(f"    {key}: {value}")
    if use_amp:
        print(f"  Mixed Precision (AMP): enabled")
    if label_smoothing > 0:
        print(f"  Label Smoothing: {label_smoothing}")
    
    print(f"  Training Transforms:")
    if train_config.get('random_resized_crop', {}).get('enabled', False):
        crop_config = train_config['random_resized_crop']
        print(f"    random_resized_crop: size={crop_config.get('size')}, scale={crop_config.get('scale')}, p={crop_config.get('p')}")
    if train_config.get('random_translate', {}).get('enabled', False):
        translate_config = train_config['random_translate']
        print(f"    random_translate: translate={translate_config.get('translate')}, p={translate_config.get('p')}")
    if train_config.get('random_rotation', {}).get('enabled', False):
        rotation_config = train_config['random_rotation']
        print(f"    random_rotation: degrees={rotation_config.get('degrees')}, p={rotation_config.get('p')}")
    if train_config.get('random_horizontal_flip', {}).get('enabled', False):
        flip_config = train_config['random_horizontal_flip']
        print(f"    random_horizontal_flip: p={flip_config.get('p')}")
    if train_config.get('random_vertical_flip', {}).get('enabled', False):
        flip_config = train_config['random_vertical_flip']
        print(f"    random_vertical_flip: p={flip_config.get('p')}")
    if train_config.get('cutmix_mixup', {}).get('enabled', False):
        cutmix_config = train_config['cutmix_mixup']
        print(f"    cutmix_mixup: alpha={cutmix_config.get('alpha')}, p={cutmix_config.get('p')}")
    
    print(f"  Test-Time Augmentations (TTA):")
    if test_config.get('resized_crop', {}).get('enabled', False):
        crop_config = test_config['resized_crop']
        print(f"    TTA: resized_crop (crop_size={crop_config.get('crop_size')}, size={crop_config.get('size')})")
    if test_config.get('fixed_affine', {}).get('enabled', False):
        affine_config = test_config['fixed_affine']
        translations = affine_config.get('translations', [[3, 3], [-3, -3]])
        print(f"    TTA: fixed_affine (translations={translations})")
    if test_config.get('random_rotation', {}).get('enabled', False):
        rotation_config = test_config['random_rotation']
        print(f"    TTA: random_rotation (degrees={rotation_config.get('degrees')})")
    if test_config.get('random_horizontal_flip', {}).get('enabled', False):
        print(f"    TTA: random_horizontal_flip")
    if test_config.get('random_vertical_flip', {}).get('enabled', False):
        print(f"    TTA: random_vertical_flip")
    
    print("=" * 80)
    
    wandb_config = {
        'dataset': dataset,
        'model': model_name,
        'pretrained': pretrained,
        'epochs': epochs,
        'batch_size': batch_size,
        'max_batch_size': max_batch_size,
        'lr': lr,
        'optimizer': optimizer_name,
        'scheduler': scheduler_name,
        'batch_schedule': batch_schedule,
        'early_stopping': early_stopping_enabled,
        'use_amp': use_amp,
        'label_smoothing': label_smoothing
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = args.config.split('/')[-1]  # Extract name from path
    
    # Track if we initialized wandb ourselves (for cleanup later)
    wandb_initialized_here = wandb.run is None
    
    # Only initialize wandb if not already initialized (e.g., by sweep.py)
    if wandb_initialized_here:
        wandb.init(
            project='carn-t3',
            config=wandb_config,
            name=f"{config_name}_{timestamp}"
        )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    prev_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        
        cutmix_mixup = get_cutmix_mixup(config, num_classes, epoch=epoch)
        
        current_lr = optimizer.param_groups[0]['lr']
        current_batch_size = train_loader.batch_size if batch_scheduler else batch_size
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp,
            train_transforms, cutmix_mixup, normalize,
            epoch=epoch, lr=current_lr, batch_size=current_batch_size, prev_val_acc=prev_val_acc
        )
        
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, use_amp, tta_transforms_list, normalize,
            epoch=epoch, lr=current_lr, batch_size=current_batch_size, prev_val_acc=prev_val_acc
        )
        
        prev_val_acc = val_acc
        
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        if batch_scheduler:
            if batch_schedule == 'plateau':
                batch_scheduler.step(metrics=val_acc)
            else:
                batch_scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': current_lr,
            'batch_size': train_loader.batch_size if batch_scheduler else batch_size
        })
        
        if early_stopping:
            if early_stopping(val_acc):
                break
    
    # Print final results for sweep script to extract
    print(f"\n{'='*80}")
    print(f"Training completed!")
    print(f"Final Validation Accuracy: {prev_val_acc:.2f}%")
    print(f"{'='*80}\n")
    
    # Only finish wandb if we initialized it ourselves (not if sweep.py initialized it)
    if wandb_initialized_here:
        wandb.finish()
    
    return prev_val_acc


if __name__ == '__main__':
    main()
