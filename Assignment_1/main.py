#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Tuple
from pathlib import Path
import numpy as np
import csv
import time
from sklearn.metrics import roc_auc_score

#paths to model and dataset
BASE_DIR   = Path("/home/atml_team038/TML ASSIGNMENT")
MODEL_PATH = BASE_DIR / "model.pt"
PUB_PATH   = BASE_DIR / "pub.pt"
PRIV_PATH  = BASE_DIR / "priv.pt"
OUT_PATH   = BASE_DIR / "submission.csv"

#configuration for the experiment 
N_SHADOW     = 64 
EPOCHS       = 75
LR           = 0.05 # used multiple, 0.05 gave best result
BATCH_SIZE   = 256
TRAIN_FRAC   = 0.5 # 50% split, every shadow model trains on half of the data 
WEIGHT_DECAY = 5e-4
SEED         = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
rng = np.random.RandomState(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


# dataset class
class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids, self.imgs, self.labels = [], [], []
        self.transform = transform

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:

        
        img = self.imgs[index]
        
        if self.transform:
            img = self.transform(img)
        return self.ids[index], img, self.labels[index]

    def __len__(self):
        return len(self.ids)


class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):

        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int, int]:
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]


def collate_fn(batch):

    
    ids        = [b[0] for b in batch]
    imgs       = torch.stack([b[1] for b in batch])

    labels     = torch.tensor([b[2] for b in batch], dtype=torch.long)
    membership = [b[3] for b in batch]
    return ids, imgs, labels, membership


MEAN = [0.7406, 0.5331, 0.7059]
STD  = [0.1491, 0.1864, 0.1301]

infer_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Normalize(mean=MEAN, std=STD),
])

# augmentation for the shadow models
train_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=2),
    transforms.Normalize(mean=MEAN, std=STD),
])


# loading data
pub_ds  = torch.load(PUB_PATH,  weights_only=False)

priv_ds = torch.load(PRIV_PATH, weights_only=False)


pub_ds.transform  = infer_transform
priv_ds.transform = infer_transform

pub_loader  = DataLoader(pub_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, pin_memory=True)
priv_loader = DataLoader(priv_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, pin_memory=True)

n_pub  = len(pub_ds)
n_priv = len(priv_ds)


pub_memberships = np.array([pub_ds[i][3] for i in range(n_pub)], dtype=int)

n_shadow_train  = int(n_pub * TRAIN_FRAC)

print(f"pub: {n_pub} samples , priv: {n_priv} samples , shadow train size: {n_shadow_train}")

#function for calculating tpr 
def tpr_at_fpr(scores, memberships, target_fpr=0.05):
    scores      = np.array(scores, dtype=float)
    memberships = np.array(memberships, dtype=int)

    members     = scores[memberships == 1]

    non_members = scores[memberships == 0]

    best_tpr = 0.0
    for thresh in np.sort(scores)[::-1]:
        if (non_members >= thresh).mean() <= target_fpr:
            best_tpr = max(best_tpr, (members >= thresh).mean())
    return best_tpr


def build_model():
    m = resnet18(weights=None)
    m.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc      = nn.Linear(512, 9)
    return m

# function for getting model confidence for each sample
@torch.no_grad()
def get_conf(model, loader):
    
    model.eval()
    out = []

    for _, imgs, labels, _ in loader:

        imgs, labels = imgs.to(device), labels.to(device)
        probs = F.softmax(model(imgs), dim=1)

        out.append(probs[torch.arange(len(labels)), labels].cpu().numpy())
    return np.concatenate(out)


# loading  target model
target_model = build_model()
target_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
target_model = target_model.to(device)
target_model.eval()

target_conf_pub  = get_conf(target_model, pub_loader)
target_conf_priv = get_conf(target_model, priv_loader)


#Lira statitics
# for public: we are  tracking confidence when each shadow model is  trained on a sample (in) vs did not (out)
# for private: every shadow model is "out" since we never train on private
conf_in_sum    = np.zeros(n_pub)
conf_out_sum   = np.zeros(n_pub)
conf_in_count  = np.zeros(n_pub,  dtype=int)
conf_out_count = np.zeros(n_pub,  dtype=int)
priv_out_sum   = np.zeros(n_priv)
priv_out_count = np.zeros(n_priv, dtype=int)

print(f"training {N_SHADOW} shadow models for ({EPOCHS} epochs each)")

for i in range(N_SHADOW):

    in_idx  = rng.choice(n_pub, size=n_shadow_train, replace=False)
    in_mask = np.zeros(n_pub, dtype=bool)
    in_mask[in_idx] = True

    pub_ds.transform = train_transform
    train_loader = DataLoader(
        Subset(pub_ds, in_idx),
        batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, pin_memory=True,
    )

    shadow    = build_model().to(device)
    optimizer = optim.SGD(shadow.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY) # got final better accuracy using sgd
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS) #for learning rate decay
    criterion = nn.CrossEntropyLoss() 

    shadow.train()
    for epoch in range(EPOCHS):

        for _, imgs, labels, _ in train_loader:

            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            criterion(shadow(imgs), labels).backward()
            optimizer.step()
        scheduler.step()

    pub_ds.transform = infer_transform

    sc_pub  = get_conf(shadow, pub_loader)
    sc_priv = get_conf(shadow, priv_loader)

    conf_in_sum[in_mask]     += sc_pub[in_mask]
    conf_in_count[in_mask]   += 1

    conf_out_sum[~in_mask]   += sc_pub[~in_mask]
    conf_out_count[~in_mask] += 1

    priv_out_sum             += sc_priv
    priv_out_count           += 1

    del shadow
    
    print(f"  shadow {i+1}/{N_SHADOW} done")


# final scores
both = (conf_in_count > 0) & (conf_out_count > 0)

mean_in  = np.where(conf_in_count  > 0, conf_in_sum  / np.maximum(conf_in_count,  1), 0.0)
mean_out = np.where(conf_out_count > 0, conf_out_sum / np.maximum(conf_out_count, 1), 0.0)

global_mean_out = conf_out_sum.sum() / np.maximum(conf_out_count.sum(), 1)

# score tells us  how much more confident was the model when this sample was in training
pub_scores = np.where(both, mean_in - mean_out, target_conf_pub - global_mean_out)

pub_tpr = tpr_at_fpr(pub_scores, pub_memberships)

pub_auc = roc_auc_score(pub_memberships, pub_scores)

print(f"\npub set: TPR@5%FPR={pub_tpr:.4f}  AUC={pub_auc:.4f}")

# for private we  compare target confidence against the shadow out-reference
mean_out_priv = priv_out_sum / np.maximum(priv_out_count, 1)
priv_scores   = target_conf_priv - mean_out_priv

# normalising to [0,1] for submission
lo, hi = priv_scores.min(), priv_scores.max()
priv_scores = (priv_scores - lo) / (hi - lo + 1e-12)

# collecting priv ids in order
priv_ids = []
for ids, _, _, _ in DataLoader(priv_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn):
    priv_ids.extend(ids)

with open(OUT_PATH, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow(["id", "score"])
    for id_, s in zip(priv_ids, priv_scores.tolist()):
        writer.writerow([id_, s])

print(f"saved {len(priv_ids)} rows -> {OUT_PATH}")
