#!/usr/bin/env python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, resnet34, resnet50
import torchvision.transforms as T
import random, os, copy

#Bockbone arechitecture
ARCH = "resnet50"

# Paths
DATA_PATH  = "/home/atml_team038/TASK_3/train.npz"
MODELS_DIR = "/home/atml_team038/TASK_3/models"
BEST_OUT   = f"/home/atml_team038/TASK_3/models/model_{ARCH}.pt"
os.makedirs(MODELS_DIR, exist_ok=True)

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# Hyperparameters
NUM_CLASSES  = 9
BATCH_SIZE   = 128
EPOCHS       = 160
LR_MAX       = 0.05
WEIGHT_DECAY = 5e-4

TRAIN_EPS   = 8 / 255.0
TRAIN_ALPHA = 2 / 255.0
TRAIN_STEPS = 10

VAL_EPS   = 8 / 255.0
VAL_ALPHA = 2 / 255.0
VAL_STEPS = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Model
ARCH_FN = {"resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50}

def build_model():
    model = ARCH_FN[ARCH](weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(DEVICE)

# dataset
class AugDataset(Dataset):
    def __init__(self, images, labels, augment=True):
        self.images, self.labels, self.augment = images, labels, augment
        self.aug = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.images[i]
        return (self.aug(img) if self.augment else img), self.labels[i]

# PGD Attack
def pgd_attack(model, x, y, eps, alpha, steps):
    was_training = model.training
    model.eval()

    x_adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1).detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)

        loss = F.cross_entropy(model(x_adv), y)

        grad = torch.autograd.grad(loss, x_adv)[0]

        x_adv = x_adv.detach() + alpha * grad.sign()

        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps).clamp(0, 1)

    if was_training:
        model.train()

    return x_adv.detach()

#evaluation function
def evaluate(model, loader):
    model.eval()

    c_correct = r_correct = total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            c_correct += (model(x).argmax(1) == y).sum().item()

            total += y.size(0)

    clean_acc = c_correct / total

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        x_adv = pgd_attack(model, x, y, VAL_EPS, VAL_ALPHA, VAL_STEPS)

        with torch.no_grad():
            r_correct += (model(x_adv).argmax(1) == y).sum().item()

    rob_acc = r_correct / total

    return clean_acc, rob_acc





data = np.load(DATA_PATH)
images = torch.from_numpy(data["images"]).float() / 255.0
labels = torch.from_numpy(data["labels"]).long()

n_val = int(0.1 * len(images))
idx = torch.randperm(len(images), generator=torch.Generator().manual_seed(SEED))
tr_idx, va_idx = idx[n_val:], idx[:n_val]

train_loader = DataLoader(
    AugDataset(images[tr_idx], labels[tr_idx], augment=True),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

val_loader = DataLoader(
    AugDataset(images[va_idx], labels[va_idx], augment=False),
    batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

model = build_model()

optimizer = optim.SGD(
    model.parameters(),
    lr=LR_MAX,
    momentum=0.9,
    weight_decay=WEIGHT_DECAY,
    nesterov=True
)

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LR_MAX,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS,
    pct_start=0.05,
    anneal_strategy="cos"
)

best_score = 0.0
best_score_state = None

best_robust_score = 0.0

best_robust_acc = 0.0

best_robust_state = None

history = []



for epoch in range(1, EPOCHS + 1):

    model.train()
    run_loss = correct = total = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        x_adv = pgd_attack(model, x, y, TRAIN_EPS, TRAIN_ALPHA, TRAIN_STEPS)

        optimizer.zero_grad()
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y, label_smoothing=0.1)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        run_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    print(f"[{ARCH}] epoch {epoch}/{EPOCHS} loss={run_loss/total:.4f} acc={correct/total:.3f}")

    #validation every 10 epochs for resnet50 because it takes mroe time 
    if epoch % 10 == 0 or epoch == EPOCHS:

        clean_acc, rob_acc = evaluate(model, val_loader)
        score = 0.5 * clean_acc + 0.5 * rob_acc

        history.append((epoch, clean_acc, rob_acc, score))

        if score > best_score:
            best_score = score
            best_score_state = copy.deepcopy(model.state_dict())

        if rob_acc > best_robust_score:
            best_robust_score = rob_acc
            best_robust_acc = rob_acc
            best_robust_state = copy.deepcopy(model.state_dict())

        print(f"validation scores: clean={clean_acc:.4f} robust={rob_acc:.4f} score={score:.4f}")

# saving models
torch.save(best_score_state, BEST_OUT)

ROBUST_OUT = BEST_OUT.replace(".pt", "_best_robust.pt")
torch.save(best_robust_state, ROBUST_OUT)


