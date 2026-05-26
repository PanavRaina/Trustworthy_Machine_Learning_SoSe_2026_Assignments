# Model Stealing: Stolen Model Detection (TML 2026)

## Recreating the best result

## Requirements: The code runs on the following docker image: uvarc/pytorch:2.0.1 

## Cluster setup

Submit file used:

```
universe		= docker
docker_image		= uvarc/pytorch:2.0.1
executable              = /home/atml_team038/tml26_task2/tml_2.py
output                  = /home/atml_team038/tml26_task2/LOGS/tml_2.$(ClusterId).$(ProcId).out
error                   = /home/atml_team038/tml26_task2/LOGS/tml_2.$(ClusterId).$(ProcId).err
log                     = /home/atml_team038/tml26_task2/LOGS/tml_2.$(ClusterId).log
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

Download the target model and the suspect models from the following link:

```
https://huggingface.co/SprintML/tml26_task2/tree/main
```

## Edit the path  
Open the `main.py` script and set the `BASE_DIR` to the directory where your files are:
```python
SUSPECT_DIR    = "/home/atml_team038/tml26_task2/suspect_models"
TARGET_WEIGHTS = "/home/atml_team038/tml26_task2/target_model/weights.safetensors"
TRAIN_IDX_JSON = "/home/atml_team038/tml26_task2/target_model/train_main_idx.json"
OUTPUT_CSV     = "submission.csv"
CIFAR_DATA_DIR = "./data" # if downloaded

```

## Run 
```bash
condor_submit shadow.sub
```

## Expected result

- Leaderboard score = 0.574074

## Output

Your output is saved in the base directory as `submission.csv`


