import json
import os
import struct
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

#used to load safetensor model wieghts. image did not have this library
_DTYPE_MAP = {
    "F32": (torch.float32, 4), "F16": (torch.float16, 2),
    "BF16": (torch.bfloat16, 2), "I64": (torch.int64, 8),
    "I32": (torch.int32, 4), "I8": (torch.int8, 1),
    "U8": (torch.uint8, 1), "BOOL": (torch.bool, 1),
}

def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n).decode("utf-8"))
        data = f.read()
    out = {}
    for k, v in meta.items():
        if k == "__metadata__":
            continue
        dtype, _ = _DTYPE_MAP[v["dtype"]]
        s, e = v["data_offsets"]
        out[k] = torch.frombuffer(bytearray(data[s:e]), dtype=dtype).reshape(v["shape"]).clone()
    return out

# resnet model definition block
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.downsample = None

        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        
        out = self.bn2(self.conv2(out))
        identity = self.downsample(x) if self.downsample else x

        return F.relu(out + identity)

class ResNet18(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.in_planes = 64
        self.conv1  = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64,  2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.fc     = nn.Linear(512, num_classes)

    def _make_layer(self, planes, n, stride):
        layers = [BasicBlock(self.in_planes, planes, stride)]

        self.in_planes = planes

        for _ in range(n - 1):
            layers.append(BasicBlock(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feat = F.adaptive_avg_pool2d(x, 1).flatten(1)
        logits = self.fc(feat)
        return logits

def load_model(path):
    model = ResNet18()
    state_dict = load_safetensors(path)
    state_dict = {k: v for k, v in state_dict.items() if "num_batches_tracked" not in k}
    model.load_state_dict(state_dict, strict=True)
    return model

# multi layer feature extractor definition
class MultiLayerExtractor:
    def __init__(self, model: nn.Module):
        self.model = model
        self.features: dict = {}
        self._hooks = []

        #hHook layer2, layer3, layer4 outputs and the  final avgpool feature
        self._hooks.append(
            model.layer2.register_forward_hook(lambda m, i, o: self.features.__setitem__("layer2", o.detach())))
        
        self._hooks.append(model.layer3.register_forward_hook(lambda m, i, o: self.features.__setitem__("layer3", o.detach())))
        self._hooks.append(model.layer4.register_forward_hook(lambda m, i, o: self.features.__setitem__("layer4", o.detach())))

    def remove(self):
        for h in self._hooks:
            h.remove()

    def get_pooled(self, key: str) -> torch.Tensor:
        
        return F.adaptive_avg_pool2d(self.features[key], 1).flatten(1)

# function for linnaer cka
def linear_CKA(X: torch.Tensor, Y: torch.Tensor) -> float:

    X = X - X.mean(0)
    Y = Y - Y.mean(0)

    XTX = X.T @ X
    YTY = Y.T @ Y
    XTY = X.T @ Y
    
    numerator   = (XTY ** 2).sum()
    denominator = torch.sqrt((XTX ** 2).sum() * (YTY ** 2).sum())
    
    if denominator < 1e-10:
        return 0.0
    return (numerator / denominator).item()

# per sample prediction and soft KL divergence function
def prediction_agreement(target_logits: torch.Tensor,suspect_logits: torch.Tensor,temperature: float = 0.5) -> tuple[float, float]:
    t_preds = target_logits.argmax(dim=1)
    s_preds = suspect_logits.argmax(dim=1)

    hard_agreement = (t_preds == s_preds).float().mean().item()

    t_probs = F.softmax(target_logits / temperature, dim=1)
    s_probs = F.softmax(suspect_logits / temperature, dim=1)

    # KL(target || suspect)
    kl = F.kl_div(s_probs.log(), t_probs, reduction="batchmean").item()
    kl_sim = 1.0 / (1.0 + max(kl, 0.0))

    return hard_agreement, kl_sim


#per-sample loss correlation

def loss_correlation(target_logits: torch.Tensor,suspect_logits: torch.Tensor,labels: torch.Tensor) -> float:

    criterion = nn.CrossEntropyLoss(reduction="none")
    t_loss = criterion(target_logits, labels)
    s_loss = criterion(suspect_logits, labels)

    # corrcoef returns 2x2 matrix
    corr_matrix = torch.corrcoef(torch.stack([t_loss, s_loss]))

    corr = corr_matrix[0, 1].item()
    return 0.0 if np.isnan(corr) else corr


def build_reference_loader(data_dir: str,all_train_indices: list,target_model: nn.Module,device: torch.device,n_reference: int = 5000,batch_size: int = 256,boundary_fraction: float = 0.5,) -> tuple[DataLoader, list]:

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.5071, 0.4867, 0.4408),std=(0.2675, 0.2565, 0.2761))
    ])
    full_train = datasets.CIFAR100(root=data_dir, train=True,download=True, transform=transform)

    #scoring candidate samples by target-model confidence
    candidate_dataset = Subset(full_train, all_train_indices)
    candidate_loader  = DataLoader(candidate_dataset, batch_size=batch_size,shuffle=False, pin_memory=True)

    confidences = []
    target_model.eval()
    with torch.no_grad():
        for images, _ in tqdm(candidate_loader, desc="Scoring candidates", leave=False):

            logits = target_model(images.to(device))
            
            probs  = F.softmax(logits, dim=1)
            
            confidences.append(probs.max(dim=1).values.cpu())
    confidences = torch.cat(confidences)           

    
    n_boundary = int(n_reference * boundary_fraction)
    n_random   = n_reference - n_boundary

    # most uncertain (lowest confidence)  ---> boundary
    boundary_local = confidences.argsort()[:n_boundary].tolist()
    # remaining indices for random sampling
    remaining = list(set(range(len(all_train_indices))) - set(boundary_local))
    rng = np.random.default_rng(42)
    random_local = rng.choice(remaining, size=min(n_random, len(remaining)),
                               replace=False).tolist()

    selected_local = boundary_local + random_local          
    selected_global = [all_train_indices[i] for i in selected_local]

    ref_dataset = Subset(full_train, selected_global)
    ref_loader  = DataLoader(ref_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)
    return ref_loader, selected_global

#collecting all signals from a model
@torch.no_grad()
def collect_signals(model: nn.Module,loader: DataLoader,device: torch.device) -> dict[str, torch.Tensor]:
    extractor = MultiLayerExtractor(model)
    model.eval()

    all_logits, all_l2, all_l3, all_l4, all_labels = [], [], [], [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)           
        all_logits.append(logits.cpu())

        all_l2.append(extractor.get_pooled("layer2").cpu())
        all_l3.append(extractor.get_pooled("layer3").cpu())
        all_l4.append(extractor.get_pooled("layer4").cpu())

        all_labels.append(labels)

    extractor.remove()
    return {
        "logits": torch.cat(all_logits, 0),
        "layer2": torch.cat(all_l2,     0),
        "layer3": torch.cat(all_l3,     0),
        "layer4": torch.cat(all_l4,     0),
        "labels": torch.cat(all_labels, 0),
    }

#computing composite similarity score
def compute_score(target_sig: dict[str, torch.Tensor],suspect_sig: dict[str, torch.Tensor],weights: dict[str, float] | None = None) -> dict[str, float]:

    if weights is None:
        weights = {
            "hard_agreement": 0.30,
            "kl_sim":         0.25,
            "cka_layer4":     0.20,
            "cka_layer3":     0.10,
            "loss_corr":      0.10,
            "cos_layer4":     0.05,
        }

    #Prediction-space metrics
    hard_agr, kl_sim = prediction_agreement(target_sig["logits"],
                                             suspect_sig["logits"])

    #CKA at layer 3 and layer 4
    cka_l3 = linear_CKA(target_sig["layer3"], suspect_sig["layer3"])
    cka_l4 = linear_CKA(target_sig["layer4"], suspect_sig["layer4"])

    #per-sample loss correlation
    loss_corr = loss_correlation(target_sig["logits"],suspect_sig["logits"],target_sig["labels"])
    
    loss_corr_01 = (loss_corr + 1.0) / 2.0

    
    cos_l4 = F.cosine_similarity(target_sig["layer4"],
                                  suspect_sig["layer4"], dim=1).mean().item()
    cos_l4_01 = (cos_l4 + 1.0) / 2.0   

    metrics = {
        "hard_agreement": hard_agr,
        "kl_sim":         kl_sim,
        "cka_layer4":     cka_l4,
        "cka_layer3":     cka_l3,
        "loss_corr":      loss_corr_01,
        "cos_layer4":     cos_l4_01,
    }

    score = sum(weights[k] * metrics[k] for k in weights)
    score = float(np.clip(score, 0.0, 1.0))
    return {"score": score, **metrics}

#paths to suspect modes, target model weights . 
SUSPECT_DIR    = "/home/atml_team038/tml26_task2/suspect_models"
TARGET_WEIGHTS = "/home/atml_team038/tml26_task2/target_model/weights.safetensors"
TRAIN_IDX_JSON = "/home/atml_team038/tml26_task2/target_model/train_main_idx.json"
OUTPUT_CSV     = "submission.csv"
CIFAR_DATA_DIR = "./data"

N_SUSPECTS        = 360
N_REFERENCE       = 8000   
BOUNDARY_FRACTION = 0.5    
BATCH_SIZE        = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}", flush=True)

#loading training indices

with open(TRAIN_IDX_JSON, "r") as f:
    all_train_indices = json.load(f)


#loading target model
print("Loading target model", flush=True)
target = load_model(TARGET_WEIGHTS).to(DEVICE).eval()


ref_loader, selected_indices = build_reference_loader(data_dir= CIFAR_DATA_DIR,all_train_indices = all_train_indices,target_model= target,device = DEVICE,n_reference= N_REFERENCE,batch_size= BATCH_SIZE,boundary_fraction = BOUNDARY_FRACTION)


#pre computing target signals
target_sig = collect_signals(target, ref_loader, DEVICE)



results = []
for model_id in tqdm(range(N_SUSPECTS), desc="Suspects"):
    path = os.path.join(SUSPECT_DIR, f"suspect_{model_id:03d}.safetensors")
    if not os.path.exists(path):
       
        results.append({"id": model_id, "score": 0.0})

        continue

    try:
        suspect = load_model(path).to(DEVICE).eval()
    except Exception as e:
        print(f"Error loading model {model_id}: {e}", flush=True)
        results.append({"id": model_id, "score": 0.0})
        continue

    # extracting all signals for the suspect
    suspect_sig = collect_signals(suspect, ref_loader, DEVICE)

    # computing score score
    result = compute_score(target_sig, suspect_sig)
    result["id"] = model_id
    results.append(result)

    
    del suspect
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


df = pd.DataFrame(results)

# keeping ONLY required columns
df = df[["id", "score"]]

#saving submission file
df.to_csv(OUTPUT_CSV, index=False)




