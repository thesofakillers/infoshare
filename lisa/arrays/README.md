# Lisa Array Scripts
In this directory you can find the scripts needed to run experiments on the Lisa cluster, using the
[job array support](https://slurm.schedmd.com/job_array.html) of the SLURM scheduler.

⚠️ **Attention:**
When scheduling a job with [sbatch](https://slurm.schedmd.com/sbatch.html), always make sure to be in the base
directory of the project (which is normally `bert-infoshare`)! Otherwise, the jobs will fail as the relative
paths for the log files will be misaligned.

⚠️ **Attention:**
Job array scripts have a unique SLURM argument, namely `--array=1-N%2` which means to read lines 1-N from the
associated hyper-parameters file and to schedule 2 jobs at a time. As such, when that file is changed (or generated
through a bash script), the corresponding line in the job script needs to be updated. If you forget to do this, then
you might not run the amount of experiments that you'd anticipate!

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

Note that you may need to provide an additional `<dir>` argument to specify
which directory to search the checkpoints in, so to disambiguate tasks that can
be run on multiple datasets, such as POS. For example:

```bash
bash lisa/arrays/get_checkpoints.sh POS lightning_logs/roberta-base/semcor
```

Then, you can schedule an array of jobs with the following command:
```bash
sbatch lisa/arrays/<task>/eval_base.job
```

## Job: Cross-neutralizing evaluation
To evaluate a set of trained models using cross-neutralizing, you first need to generate the pairs of
(checkpoint, neutralizer tag).

You can do this automatically with the following bash script:
```bash
bash lisa/arrays/generate_neutr_hparams.sh <task>
```

Then, you can schedule an array of jobs with the following command:
```bash
sbatch lisa/arrays/<task>/eval_xneutral.job
```

## Job: Cross-lingual cross-neutralizing evaluation
To evaluate a set of trained models using cross-lingual cross-neutralizing, you first need to generate
the pairs of (neutralizer_checkpoint, target_checkpoint, neutralizer_tag).

You can do this automatically with the following bash script:
```bash
bash lisa/arrays/generate_xlingual_neutr_hparams.sh <task>
```
This script assumes that you have a `cherry_checkpoints.txt` file in the `lisa/arrays/<task>` directory,
which contains the list of checkpoints to generate all (neutralizer, target) pairs with.

Then, you can schedule an array of jobs with the following command:
```bash
sbatch lisa/arrays/<task>/eval_xneutral_xlingual.job
```
