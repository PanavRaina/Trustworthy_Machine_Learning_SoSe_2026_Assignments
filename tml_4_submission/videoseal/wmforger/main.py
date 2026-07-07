import os
import sys
import shutil
import numpy as np
import torch
import omegaconf
import torchvision
import cv2
from PIL import Image
from tqdm import tqdm

# paths to files 
CKPT_PATH = "./convnext_pref_model.pth"
CONFIG_PATH = "./configs/extractor.yaml"
CLEAN_DIR = "../../Dataset/clean_targets"
WM_ROOT = "../../Dataset/watermarked_sources"
OUTPUT_DIR = "./forged_final_raftaar12345"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#mapping from watermark group to clean image ID range
GROUP_RANGES = {
    "WM_1": (1, 25),
    "WM_2": (26, 50),
    "WM_3": (51, 75),
    "WM_4": (76, 100),
    "WM_5": (101, 125),
    "WM_6": (126, 150),
    "WM_7": (151, 175),
    "WM_8": (176, 200),
}

# groups to be analysed seperately 
LIBRARY_OVERRIDES = {
    "WM_1": {
        "method": "dwtDct",
        "n_bits": 16,
        "message": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
    },
    "WM_2": {
        "method": "rivaGan",
        "n_bits": 32,
        "message": [
            0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1,
            0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0,
        ],
    },
}

# model based groups
NEURAL_GROUPS = ["WM_3", "WM_4", "WM_5", "WM_6", "WM_7", "WM_8"]
NUM_STEPS = 50
LR = 0.05
L2_PENALTY = 0.05
RESOLUTION = 768
STRENGTH = 1.0                 # watermark strength
N_REFERENCE_IMAGES = 25        # number of watermarked images to average over

# ensure imports 
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "videoseal"))
from wmforger.models import build_extractor


transform_image = torchvision.transforms.Compose([
    lambda x: x.convert("RGB"),
    torchvision.transforms.Resize((RESOLUTION, RESOLUTION)),
    torchvision.transforms.ToTensor(),
    lambda x: x.view(1, 3, RESOLUTION, RESOLUTION),
])

# ---------------------------------------------------------

#loaidng model
def load_model():
   
    model_type = "convnext_tiny"
    state_dict = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)["model"]
    extractor_params = omegaconf.OmegaConf.load(CONFIG_PATH)[model_type]
    model = build_extractor(model_type, extractor_params, img_size=256, nbits=0)
    model.load_state_dict(state_dict)
    return model.eval().to(DEVICE)


def extract_watermark(img_path, model, desc=""):

    # Load and preprocess image
    img = Image.open(img_path).convert("RGB")


    orig_size = img.size
    x = transform_image(img).to(DEVICE)

   
    delta = torch.nn.Parameter(torch.zeros_like(x))
    optim = torch.optim.SGD([delta], lr=LR)


    for _ in tqdm(range(NUM_STEPS), desc=f"Extracting {desc}", leave=False):
        optim.zero_grad()

        score = model((x + delta).clip(0, 1)).mean()
        
        loss = -score + L2_PENALTY * delta.pow(2).mean()
        
        loss.backward()
        optim.step()

    
    with torch.no_grad():
        optimised = (x + delta).clip(0, 1).cpu()

        
        
        optimised_np = optimised.mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).squeeze(0).numpy()
        
        optimised_img = Image.fromarray(optimised_np).resize(orig_size, Image.BILINEAR)
        watermark = np.array(img).astype(np.float32) - np.array(optimised_img).astype(np.float32)

    return watermark


def resize_watermark(wm, target_h, target_w):

    shifted = np.clip(wm + 128, 0, 255).astype(np.uint8)
    resized = Image.fromarray(shifted).resize((target_w, target_h), Image.BILINEAR)

    return np.array(resized).astype(np.float32) - 128


def forge_using_library(group, cfg, clean_ids):

    
    encoder_mod = __import__("imwatermark", fromlist=["WatermarkEncoder"])
    WatermarkEncoder = encoder_mod.WatermarkEncoder

    if cfg["method"] == "rivaGan":
        WatermarkEncoder.loadModel()   # load pre‑trained RivaGAN model

    encoder = WatermarkEncoder()
    encoder.set_watermark("bits", cfg["message"])

    written = 0
    for img_id in clean_ids:

        clean_path = os.path.join(CLEAN_DIR, f"{img_id}.png")
        if not os.path.exists(clean_path):
            continue

        bgr = cv2.imread(clean_path)
        orig_h, orig_w = bgr.shape[:2]

        
        if min(orig_h, orig_w) < 256:
            scale = 256.0 / min(orig_h, orig_w) * 1.05
            bgr_work = cv2.resize(bgr, (int(orig_w * scale), int(orig_h * scale)))
        else:
            
            bgr_work = bgr

        # encode watermark
        bgr_enc = encoder.encode(bgr_work, cfg["method"])

        
        
        if min(orig_h, orig_w) < 256:
            bgr_enc = cv2.resize(bgr_enc, (orig_w, orig_h))

        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{img_id}.png"), bgr_enc)
        written += 1

    return written


def forge_using_neural(group, model, clean_ids):
    
    # collect watermarks from a subset of reference images in the group
    
    folder = os.path.join(WM_ROOT, group)
    files = sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])[:N_REFERENCE_IMAGES]

    watermarks = []
    target_shape = None
    
    for file_path in files:
        wm = extract_watermark(file_path, model, desc=os.path.basename(file_path))
        
        if target_shape is None:
            target_shape = wm.shape[:2]
        elif wm.shape[:2] != target_shape:
            wm = resize_watermark(wm, *target_shape)
        watermarks.append(wm)

    
    avg_wm = np.mean(watermarks, axis=0)
    
    # Apply the averaged watermark to each clean image
    written = 0
    for img_id in clean_ids:
        clean_path = os.path.join(CLEAN_DIR, f"{img_id}.png")
        if not os.path.exists(clean_path):
            continue

       
        clean_img = Image.open(clean_path).convert("RGB")
        
        clean_np = np.array(clean_img).astype(np.float32)
        h, w = clean_np.shape[:2]

        wm_resized = resize_watermark(avg_wm, h, w)
        forged_np = np.clip(clean_np + STRENGTH * wm_resized, 0, 255).astype(np.uint8)
       
        Image.fromarray(forged_np).save(os.path.join(OUTPUT_DIR, f"{img_id}.png"))
        written += 1

    return written



    
os.makedirs(OUTPUT_DIR, exist_ok=True)

for group, cfg in LIBRARY_OVERRIDES.items():
    start, end = GROUP_RANGES[group]
    print(f"\n=== {group}: using library method '{cfg['method']}' ===")
    count = forge_using_library(group, cfg, range(start, end + 1))
    print(f"  Written {count} images")

#model based
if NEURAL_GROUPS:
    model = load_model()
    for group in NEURAL_GROUPS:
        start, end = GROUP_RANGES[group]

        print(f"\n=== {group}: using model method ===")
        
        count = forge_using_neural(group, model, list(range(start, end + 1)))
        print(f"  Written {count} images")
        torch.cuda.empty_cache()



shutil.make_archive("submission", "zip", OUTPUT_DIR)
print("Created submission.zip")

