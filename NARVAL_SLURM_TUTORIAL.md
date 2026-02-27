# Narval (Calcul Québec) Tutorial with Slurm

This guide explains how to install this project and run training jobs on
Narval.

## 1) Prerequisites

-   An active Alliance/Calcul Québec account.
-   A Slurm account/project (`--account=...`, e.g., `def-monpi`).
-   SSH access to Narval.

## 2) Where to place the files

Practical recommendation: 
- code + temporary checkpoints: `$SCRATCH`
- final results to keep: `$PROJECT` (or other persistent storage)
- dataset: for better I/O performance (especially with many small files),
you can temporarily copy the dataset to `$SLURM_TMPDIR`, which
corresponds to the local storage of the compute node allocated to the
job; its content is automatically deleted at the end of the job.

Note: The `$SCRATCH` directory is subject to an automatic purge policy
(inactive files are deleted after some time); it must not be used for
long-term storage. Important outputs should be copied to `$PROJECT` or
another persistent storage location.

Example:

``` bash
cd $SCRATCH
git clone https://github.com/Leon-afk1/NoduLoCC2026.git
cd NoduLoCC2026
git checkout leonard
```

Copy the dataset to Narval:

``` bash
rsync -avz local_path user@narval.calculquebec.ca:~/scratch/
```

Create a tar archive of the dataset to simplify staging inside the Slurm
job:

``` bash
cd $SCRATCH
tar -cf nodulocc_dataset.tar nodulocc_dataset/
```

## 3) Prepare the Python environment (on login node)

``` bash
cd $SCRATCH/NoduLoCC2026
module load python/3.14.2 # adapt to what "module avail python" shows
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .[exp]
```

## 4) Quick check before batch

``` bash
cd $SCRATCH/NoduLoCC2026
source .venv/bin/activate
python -m nodulocc.cli --help
```

## 5) Slurm GPU script

The repository contains a template: `train_gpu.slurm.example`

Copy the template and edit the required fields (Slurm account, email):

``` bash
cd $SCRATCH/NoduLoCC2026
cp train_gpu.slurm.example train_gpu.slurm
nano train_gpu.slurm   # or vim
```

Submit:

``` bash
sbatch train_gpu.slurm
```

To automatically submit the job and synchronize W&B after completion,
use the `submit_and_sync.sh` script:

``` bash
./submit_and_sync.sh
```

## 6) Useful Slurm commands

``` bash
squeue -u $USER # view queued/running jobs
sq # quick alias for squeue -u $USER
sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,MaxRSS # details of a finished job
scancel <JOBID> # cancel a queued or running job
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS,ReqMem # view resources used by a finished job
```

## 7) K-fold and final training

Modify the configuration or use overrides to perform K-fold
cross-validation or train on 100% of the data.

### K-fold

``` bash
python -m nodulocc.cli train --config configs/classification.yaml \
    --override data.dataset_root=$SLURM_TMPDIR/nodulocc_dataset \
    --override validation.mode=kfold \
    --override validation.k=5
```

### Final training on 100% of the data

``` bash
python -m nodulocc.cli train --config configs/classification.yaml \
   --override data.dataset_root=$SLURM_TMPDIR/nodulocc_dataset \
   --override train.full_data=true
```

## 8) W&B on Narval (important)

According to Alliance documentation, on Narval it is recommended to use
`wandb` in offline mode during compute jobs.

Inside the job:

``` bash
export WANDB_MODE=offline
```

After the job (on login node), synchronize:

``` bash
cd $SCRATCH/NoduLoCC2026
source .venv/bin/activate
wandb sync --sync-all
```

## 9) HPC best practices

-   Request realistic resources (`time`, `mem`, `cpus`, `gpu`) to reduce
    queue time.
-   Write checkpoints/logs to `$SCRATCH` during the run.
-   Copy important final results to `$PROJECT` at the end of the
    campaign.
-   Version control the run configuration (`configs/*.yaml` + overrides)
    along with checkpoints.

## 10) Sources

-   Alliance Doc: W&B on clusters and offline mode.
-   Alliance Doc: Slurm/job script principles.

Cluster policies, modules, and quotas may evolve: always check the
Alliance/Calcul Québec documentation before running a large campaign.
