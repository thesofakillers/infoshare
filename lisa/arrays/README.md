# Lisa Array Scripts
In this directory you can find the scripts needed to run experiments on the Lisa cluster, using the
[job array support](https://slurm.schedmd.com/job_array.html) of the SLURM scheduler.

⚠️ **Attention:**
When scheduling a job with [sbatch](https://slurm.schedmd.com/sbatch.html), always make sure to be in the base
directory of the project (which should usually be `bert-infoshare`)! Otherwise, the jobs will fail as the relative
paths for the log files will be misaligned.

## Job: Grid search for training
To perform grid search for training, after modifying the file [hparams.txt](hparams.txt) accordingly, you can run
the following command:
```bash
sbatch lisa/arrays/<task>/train.job
```
where `<task>` is either **pos** or **dep** (similarly for all occurences in the following examples).

## Job: Baseline evaluation
To evaluate a set of trained models, you first need to define the corresponding checkpoints.

You can do this automatically with the following bash script:
```bash
bash lisa/arrays/get_checkpoints.sh <task>
```

Then, you can schedule an array of jobs with the following command:
```bash
sbatch lisa/arrays/<task>/eval_base.job
```

## Job: (Cross-) Neutralizing evaluation
To evaluate a set of trained models using (cross-)neutralizing, you first need to generate the pairs of
(checkpoint, neutralizer tag).

You can do this automatically with the following bash script:
```bash
bash lisa/arrays/generate_neutralizer_tags.sh <task>
```

Then, you can schedule an array of jobs with the following command:
```bash
sbatch lisa/arrays/<task>/eval_xneutral.job
```