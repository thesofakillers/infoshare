# What is in this folder?

We collect our [jupyter](https://jupyter.org/) notebooks here. Please refer to
the following table for what each notebook contains:

| **Notebook name**           | **Contents description**                                                                                                                                                                |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cross_neutralizing.ipynb`  | Code for plotting the cross-neutralization heatmaps for every possible combination of POS/DEP, en_gum/it_vit/el_gdt, and roberta-base/xlm-roberta-base.                                 |
| `cross_task_figs.ipynb`     | Code for plotting the heatmaps from our cross-task cross-neutralization experiment with roberta-base, where we neutralizing the classes for one task using centroids from another task. |
| `cross_task_figs_xlm.ipynb` | The same as `cross_task_figs.ipynb`, but for XLM-R and the associated en, it, el treebanks.                                                                                             |
| `infer_demo.ipynb`          | Quick inference demo from the start of the project                                                                                                                                      |
| `lswsd_lemmas.ipynb`        | Searching for candidate words for a lexical-sample word-sense disambiguation experiment that we did not ultimately pursue.                                                              |
| `lswsd_xtask.ipynb`         | Attempt at performing cross-task neutralization of LSWSD classes using POS centroids. Ultimately abandoned due to insufficient data.                                                    |
| `report_figures.ipynb`      | Collating all the code for the figures in the report into a single notebook. Partially stale as of 03/2023.                                                                             |
| `semcor_sense_stats.ipynb`  | Brief analysis of sense statistics in the SemCor dataset                                                                                                                                |
| `semcor_word_stats.ipynb`   | Exploring the SemCor dataset.                                                                                                                                                           |
| `ud_stats.ipynb`            | Exploring the Universal Dependencies dataset.                                                                                                                                           |
