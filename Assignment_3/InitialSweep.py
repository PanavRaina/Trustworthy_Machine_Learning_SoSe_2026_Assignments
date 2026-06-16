#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, resnet34, resnet50
import torchvision.transforms as T
import random, os, csv, shutil

# Paths 
DATA_PATH   = "/home/atml_team038/TASK_3/train.npz"
MODELS_DIR  = "/home/atml_team038/models"
BEST_OUT    = "/home/atml_team038/model.pt"
CSV_OUT     = os.path.join(MODELS_DIR, "sweep_results.csv")

os.makedirs(MODELS_DIR, exist_ok=True)

#for reeproducibility 
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# hyperparameters
NUM_CLASSES  = 9
BATCH_SIZE   = 128
EPOCHS       = 90       
LR_MAX       = 0.1
WEIGHT_DECAY = 5e-4
LABEL_SMOOTH = 0.1
ALPHA_RATIO  = 0.25        

# validation parameters

VAL_EPS      = 8  / 255.0
VAL_ALPHA    = 2  / 255.0
VAL_STEPS    = 20


ARCHITECTURES = ["resnet18","resnet34", "resnet50"]
EPS_VALUES    = [4, 8]          
STEP_VALUES   = [7, 10, 14]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

total_runs = len(ARCHITECTURES) * len(EPS_VALUES) * len(STEP_VALUES)



ARCH_FN = { "resnet18":resnet18,"resnet34": resnet34, "resnet50": resnet50}

def build_model(arch_name):
    model = ARCH_FN[arch_name](weights=None)

    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    return model.to(DEVICE)

def model_tag(arch, eps_int, steps):
    # naming the model based on the paramters used
    return f"{arch}_eps{eps_int}_steps{steps}"

class AugDataset(Dataset):
    def __init__(self, images, labels, augment=True):
        self.images = images

        self.labels = labels


        self.aug = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
        ]) if augment else None

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        img = self.images[i]
        return (self.aug(img) if self.aug else img), self.labels[i]

def pgd_attack(model, x, y, eps, alpha, steps):
    model.eval()
    x_adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)

        loss = nn.CrossEntropyLoss()(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()

        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps).clamp(0, 1)
    model.train()
    return x_adv.detach()

# evaluationc code
def evaluate(model, loader):
    
    model.eval()
    
    c_correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            c_correct += (model(x).argmax(1) == y).sum().item()
            total     += y.size(0)
    clean_acc = c_correct / total

    # robustness
    r_correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        x_adv = pgd_attack(model, x, y, VAL_EPS, VAL_ALPHA, VAL_STEPS)
        with torch.no_grad():
            r_correct += (model(x_adv).argmax(1) == y).sum().item()
        
        total += y.size(0)
    rob_acc = r_correct / total

    return clean_acc, rob_acc


def run_experiment(arch_name, eps_int, pgd_steps, train_loader, val_loader):

    tag   = model_tag(arch_name, eps_int, pgd_steps)
    eps   = eps_int / 255.0
    alpha = eps * ALPHA_RATIO
    save_path = os.path.join(MODELS_DIR, f"{tag}.pt")

    
    print(f"[run] {tag}  eps={eps:.5f}  alpha={alpha:.5f}  steps={pgd_steps}", flush=True)
    

    model     = build_model(arch_name)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    optimizer = optim.SGD(model.parameters(), lr=LR_MAX,momentum=0.9, weight_decay=WEIGHT_DECAY, nesterov=True)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr          = LR_MAX,
        steps_per_epoch = len(train_loader),
        epochs          = EPOCHS,
        pct_start       = 0.05,
        anneal_strategy = "cos",
    )

    best_score     = 0.0
    best_clean     = 0.0
    best_robust    = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        run_loss, correct, total = 0.0, 0, 0

        for x, y in train_loader:
            x, y  = x.to(DEVICE), y.to(DEVICE)
            x_adv = pgd_attack(model, x, y, eps, alpha, pgd_steps)

            model.train()
            optimizer.zero_grad()

            logits = model(x_adv)
            loss   = criterion(logits, y)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            run_loss += loss.item() * y.size(0)
            correct  += (logits.argmax(1) == y).sum().item()
            total    += y.size(0)

        print(f"[{tag}] epoch {epoch:3d}/{EPOCHS}  "f"loss={run_loss/total:.4f}  adv_acc={correct/total:.3f}", flush=True)

        # validation score check after every 5 epochs
        if epoch % 25 == 0 or epoch == EPOCHS:
            
            clean_acc, rob_acc = evaluate(model, val_loader)
            score = 0.5 * clean_acc + 0.5 * rob_acc
            print(f"[{tag}] validation scores:  clean={clean_acc:.4f}  "f"robust={rob_acc:.4f}  score={score:.4f}", flush=True)

            if score > best_score:
                best_score  = score
                best_clean  = clean_acc

                best_robust = rob_acc

                torch.save(model.state_dict(), save_path)
                print(f"[{tag}] saved at {save_path}", flush=True)

    print(f"[{tag}] FINAL  clean={best_clean:.4f}  "
          f"robust={best_robust:.4f}  score={best_score:.4f}", flush=True)

    return {
        "model_name" : tag,
        "arch"       : arch_name,
        "eps_int"    : eps_int,
        "pgd_steps"  : pgd_steps,
        "clean_acc"  : round(best_clean,  4),
        "robust_acc" : round(best_robust, 4),
        "score"      : round(best_score,  4),
        "save_path"  : save_path,
    }



# Load data once
print(f"[data] loading {DATA_PATH}", flush=True)
data   = np.load(DATA_PATH)
images = torch.from_numpy(data["images"]).float() / 255.0
labels = torch.from_numpy(data["labels"]).long()
print(f"[data] {len(images)} images  labels {labels.min()}–{labels.max()}", flush=True)

# 90/10 split, fixed across all runs for fair comparison
n_val   = int(0.1 * len(images))
idx     = torch.randperm(len(images), generator=torch.Generator().manual_seed(SEED))
tr_idx, va_idx = idx[n_val:], idx[:n_val]

train_loader = DataLoader(
    AugDataset(images[tr_idx], labels[tr_idx], augment=True),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(
    AugDataset(images[va_idx], labels[va_idx], augment=False),
    batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

results = []
run_num = 0
for arch in ARCHITECTURES:
    for eps_int in EPS_VALUES:
        for steps in STEP_VALUES:
            run_num += 1
            print(f"\n[sweep] run {run_num}/{total_runs}", flush=True)
            result = run_experiment(arch, eps_int, steps, train_loader, val_loader)
            results.append(result)

#sweep results 
print("Sweep Result:", flush=True)
print(f" {'model':<35} {'clean':>7} {'robust':>7} {'score':>7}", flush=True)
for r in sorted(results, key=lambda x: -x["score"]):
    print(f"  {r['model_name']:<35} "f"{r['clean_acc']:>7.4f} {r['robust_acc']:>7.4f} "f"{r['score']:>7.4f}", flush=True)




