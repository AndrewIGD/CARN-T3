import os

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision.transforms import v2
from torch.backends import cudnn
from torch import GradScaler
from torch import optim
from tqdm import tqdm
import numpy as np
import pickle
import os

#############################################
#               Muon optimizer              #
#############################################

@torch.compile
def zeropower_via_newtonschulz5(G, steps=3, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Works with 2D matrices - gradients are reshaped to 2D for Conv2D layers.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    # Use float32 for numerical stability - the overhead is minimal compared to the full training
    X = G.float()  # Keep in float32 to avoid numerical issues
    X /= (X.norm() + eps)  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9, nesterov=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                g = p.grad
                state = self.state[p]
                
                # Initialize momentum buffer
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                
                # Apply Nesterov momentum if enabled
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                
                # Normalize the weight
                p.data.mul_(len(p.data)**0.5 / p.data.norm())
                
                # Reshape gradient to 2D and apply Newton-Schulz whitening
                # For Conv2D: (out, in, h, w) -> (out, in*h*w)
                # For Linear: (out, in) -> (out, in) [no change]
                flat_g = g.reshape(len(g), -1)
                update = zeropower_via_newtonschulz5(flat_g).view(g.shape)
                
                # Take gradient step
                p.data.add_(update, alpha=-lr)

device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device("cpu")
enable_half = device.type != "cpu"
scaler = GradScaler(device, enabled=enable_half)

print(torch.__version__)

print("Grad scaler is enabled:", enable_half)
device

if os.path.exists("/kaggle/input") and os.path.exists("/kaggle/working"):
    print("Running on Kaggle.")
    SVHN_test = "/kaggle/input/fii-atnn-2025-competition-2/SVHN_test.pkl"
    SVHN_train = "/kaggle/input/fii-atnn-2025-competition-2/SVHN_train.pkl"
else:
    print("Not on Kaggle.")
    SVHN_test = "data/SVHN_test.pkl"
    SVHN_train = "data/SVHN_train.pkl"

class SVHN_Dataset(Dataset):
    def __init__(self, train: bool, transforms: v2.Transform):
        path = SVHN_test
        if train:
            path = SVHN_train
        with open(path, "rb") as fd:
            self.data = pickle.load(fd)

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int):
        image, label = self.data[i]
        if self.transforms is None:
            return image, label
        return self.transforms(image), label

num_outputs = 100

random_crop = v2.RandomApply([v2.RandomResizedCrop(size=(32, 32), scale=(0.7, 1.0))], p=0.4)
random_translate = v2.RandomApply([v2.RandomAffine(degrees=0, translate=(0.1, 0.2))], p=0.4)
random_flipping = v2.RandomHorizontalFlip(p=0.4)

# Will be created dynamically with epoch-specific alpha
cutmix = None
mixup = None
cutmix_or_mixup = None

crop = v2.RandomResizedCrop(size=(32, 32), scale=(0.8, 1.0))
translate = v2.RandomAffine(degrees=0, translate=(0.05, 0.15))
flipping = v2.RandomHorizontalFlip(p=1)

cpu_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

train_transforms = v2.Compose([
    random_crop,
    random_translate,
    random_flipping,
])

tta_crop = v2.Compose([
    v2.CenterCrop(size=(28)),
    v2.Resize(size=(32, 32))
    
])
tta_flip = v2.RandomHorizontalFlip(p=1)

class FixedAffine:
    """Deterministic affine transform with fixed parameters"""
    def __init__(self, dx=3, dy=3):
        self.dx = dx
        self.dy = dy
    
    def __call__(self, img):
        return v2.functional.affine(img, angle=0, translate=(self.dx, self.dy), scale=1.0, shear=0)

tta_translate_p3 = FixedAffine(dx=3, dy=3)
tta_translate_m3 = FixedAffine(dx=-3, dy=-3)

# Normalization transform (applied after augmentations)
normalize = v2.Normalize((0.4786564111709595, 0.4788946509361267, 0.4769909679889679), (0.2681833505630493, 0.2682854235172272, 0.26810330152511597), inplace=True)

# TTA transforms wrapped with normalization
tta_transforms_list = [
    v2.Compose([tta_crop, normalize]),
    v2.Compose([tta_flip, normalize]),
    v2.Compose([tta_translate_p3, normalize]),
    v2.Compose([tta_translate_m3, normalize])
]

train_set = SVHN_Dataset(train=True, transforms=cpu_transforms)
test_set = SVHN_Dataset(train=False, transforms=cpu_transforms)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=500)

class VGG13(nn.Module):
    def __init__(self):
        super(VGG13, self).__init__()

        self.layers = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Classifier
            nn.Flatten(),
            nn.Linear(512, num_outputs)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

model = VGG13().to(device)

muon_params = [p for p in model.parameters() if p.ndim == 4]
sgd_params = [p for p in model.parameters() if p.ndim < 4]

optimizer1 = optim.SGD(sgd_params, lr=0.005, momentum=0.85, nesterov=True, fused=True)
optimizer2 = Muon(muon_params, lr=0.005, momentum=0.8, nesterov=True)

optimizers = [optimizer1, optimizer2]

model = torch.jit.script(model)
criterion = nn.CrossEntropyLoss()

# Learning rate schedulers - reduce LR when accuracy reaches ~85%
scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer1, mode='max', factor=0.85, patience=4, 
    threshold=0.01, threshold_mode='rel', cooldown=2, min_lr=1e-6
)
scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer2, mode='max', factor=0.85, patience=4, 
    threshold=0.01, threshold_mode='rel', cooldown=2, min_lr=1e-6
)

def train(epoch):
    model.train()
    correct = 0
    total = 0
    
    # Calculate alpha: ramp from 0 to 0.75 over first 25 epochs
    if epoch < 25:
        alpha = 0.75 * (epoch / 25.0)
    else:
        alpha = 0.75
    
    # Create CutMix/MixUp transforms with current alpha
    current_cutmix = v2.CutMix(num_classes=num_outputs, alpha=alpha)
    current_mixup = v2.MixUp(num_classes=num_outputs, alpha=alpha)
    current_cutmix_or_mixup = v2.RandomChoice([current_cutmix, current_mixup])
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        inputs = train_transforms(inputs)
        
        # Apply CutMix/MixUp
        inputs, targets = current_cutmix_or_mixup(inputs, targets)
        
        inputs = normalize(inputs)
        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        for optimizer in optimizers:
            scaler.step(optimizer)
            optimizer.zero_grad()
        scaler.update()

        predicted = outputs.argmax(1)
        total += targets.size(0)
        if targets.dim() > 1:
            target_classes = targets.argmax(1)
        else:
            target_classes = targets
        correct += predicted.eq(target_classes).sum().item()
    
    return 100.0 * correct / total

@torch.inference_mode()
def inference():
    model.eval()
    
    all_predictions = []
    
    for inputs, _ in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        
        batch_predictions = []
        
        # Normalize and run base prediction
        normalized_inputs = normalize(inputs.clone())
        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(normalized_inputs)
        batch_predictions.append(outputs)
        
        for transform in tta_transforms_list:
            augmented_inputs = transform(inputs)
            with torch.autocast(device.type, enabled=enable_half):
                outputs = model(augmented_inputs)
            batch_predictions.append(outputs)
        
        batch_predictions = torch.stack(batch_predictions)
        mean_predictions = batch_predictions.mean(dim=0)
        all_predictions.append(mean_predictions.argmax(dim=1))
    
    final_predictions = torch.cat(all_predictions, dim=0)
    return final_predictions.cpu().numpy()

best = 0.0
best_epoch = 0
epochs = list(range(300))

for epoch, _ in enumerate(epochs):
    train_acc = train(epoch)

    scheduler1.step(train_acc)
    scheduler2.step(train_acc)
        
    current_lr1 = optimizer1.param_groups[0]['lr']
    current_lr2 = optimizer2.param_groups[0]['lr']

    if train_acc > best:
        best = train_acc
        best_epoch = epoch

    print(f"Epoch {epoch}: Train: {train_acc:.2f}, Best: {best:.2f} at epoch {best_epoch}, LR1: {current_lr1:.5f}, LR2: {current_lr2:.5f}")

data = {
    "ID": [],
    "target": []
}

for i, label in enumerate(inference()):
    data["ID"].append(i)
    data["target"].append(label)

df = pd.DataFrame(data)

on_kaggle = os.path.exists("/kaggle/input") and os.path.exists("/kaggle/working")
if on_kaggle:
    submission_path = f"/kaggle/working/submission.csv"
else:
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    submission_path = f"results/submission_{script_name}.csv"

df.to_csv(submission_path, index=False)