# Watermark Forgery Attack  (TML 2026)

## Recreating the best result

## Requirements: Install all the required libraries mentioned in the requirements.txt file 
```bash
pip install -r requirements.txt
```

## The file to run is `./videoseal/wmforger/main.py`. This file will give our result
## The analysis.py is our other experiment file . It can be found in  `./videoseal/wmforger/analysis.py`

## Before running the code, download the model using the following command 
```
wget https://dl.fbaipublicfiles.com/wmforger/convnext_pref_model.pth
```
Place the model in the following directory: `./videoseal/wmforger/`

## Data files 

Download the dataset 


## Edit the path  
Open the `main.py` script and provide the following paths:
```python
CKPT_PATH = "./convnext_pref_model.pth" # this path should be correct if you download it in the correct directory, so no need to change
CONFIG_PATH = "./configs/extractor.yaml" # this path is should be correct as its relative path
CLEAN_DIR = "../../Dataset/clean_targets" # provide path to you clean targets folder of the dataset
WM_ROOT = "../../Dataset/watermarked_sources" # provide path to you watermarked sources folder of the dataset
OUTPUT_DIR = "./forged_final_raftaar12345"
```

## Run 
run this from inside `tml_4_submission/videoseal/wmforger/` directory as CKPT_PATH and CONFIG_PATH paths are relative to that directory.

## Expected result

- Leaderboard: 0.538619

## Output

The output is saved in the  `./videoseal/wmforger/` directory as `submission.zip`


