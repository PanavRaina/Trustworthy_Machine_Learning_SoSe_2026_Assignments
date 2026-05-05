# Membership Inference Attack (TML 2026)

## Recreating the best result

## Requirements: The code runs on the following docker image: uvarc/pytorch:2.0.1 and additoionally requires the following python package: scikit-learn

## Cluster setup

Submit file used:

```
universe		= docker
docker_image		= uvarc/pytorch:2.0.1
executable              = /home/atml_team038/assignment/shadow.py
output                  = /home/atml_team038/shadow.$(ClusterId).$(ProcId).out
error                   = /home/atml_team038/shadow.$(ClusterId).$(ProcId).err
log                     = /home/atml_team038/shadow.$(ClusterId).log
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

Download the data files and the model using the following commands:

```
wget "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/pub.pt"
wget "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/priv.pt"
wget "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/model.pt"
```

## Edit the path  
Open the `main.py` script and set the `BASE_DIR` to the directory where your files are:
```python
BASE_DIR = Path("/your/path/here")
```

## Run 
```bash
condor_submit shadow.sub
```

## Expected result

- Public set:  TPR @ 5% FPR = 0.0546
- Private leaderboard: TPR @ 5% FPR = 0.0598

## Output

Your output is saved in the base directory as `submission.csv`


