# Model Stealing: Stolen Model Detection (TML 2026)

## Recreating the best result

## To recreate the best result, run the ResNet50.py file

## Requirements: The code runs on the following docker image: uvarc/pytorch:2.0.1 

## Cluster setup

Submit file used:

```
universe		= docker
docker_image		= uvarc/pytorch:2.0.1
executable              = /home/atml_team038/TASK_3/ResNet50.py
output                  = /home/atml_team038/TASK_3/LOGS/ResNet50.$(ClusterId).$(ProcId).out
error                   = /home/atml_team038/TASK_3/LOGS/ResNet50.$(ClusterId).$(ProcId).err
log                     = /home/atml_team038/TASK_3/LOGS/ResNet50.$(ClusterId).log
should_transfer_files   = YES
request_GPUs = 1
request_CPUs = 1
request_memory = 8G
getenv = HOME
requirements = UidDomain == "cs.uni-saarland.de" 
+WantGPUHomeMounted = true
+WantScratchMounted = true
queue 1

```

## Data files and model

Download the dataset from the following link:

```
https://huggingface.co/datasets/SprintML/tml26_task3/blob/main/train.npz
```

## Edit the path  
Open the `ResNet50.py` script and set the following paths to the directory where your files are:
```python
DATA_PATH  = "/home/atml_team038/TASK_3/train.npz"
MODELS_DIR = "/home/atml_team038/TASK_3/models"
BEST_OUT   = f"/home/atml_team038/TASK_3/models/model_{ARCH}.pt"

```

## Run 
```bash
condor_submit main.sub
```

## Expected result

- Leaderboard score = 0.604251



