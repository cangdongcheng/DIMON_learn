# DIffeomorphic-Mapping Operator learNing (DIMON)

We introduce a neural operator framework, named **DIffeomorphic Mapping Operator learNing (DIMON)**, which allows AI to learn geometry-dependent solution operators of different types of PDEs on a wide variety of geometries. Nature Computational Science: https://www.nature.com/articles/s43588-024-00732-2

![dimon](./figure/dimon.png)

**Data**: https://doi.org/10.5281/zenodo.13958884

**Note**: Data in example 3 is not included in this link as the heart geometries are patient-specific clinical data. Data in Example 3 can be provided by request to the corresponding authors and potentially after an IRB for sharing data is approved.

## Environment (HPC)

The upstream README assumes a personal laptop with `python -m venv` + `pip install`. On this HPC the recommended workflow is to use the system's `pytorch` module, which already provides torch + numpy + scipy + matplotlib for the right Python and CUDA. Only `scikit-learn` needs a one-time user install.

**One-time setup** (already done on this account):
```bash
module load pytorch/2.6.0-py3-cu11.8
pip install --user scikit-learn
```
This installs `scikit-learn` into `~/.local/lib/python3.13/site-packages`, alongside the module's site-packages, and is picked up automatically whenever the module is loaded.

**Every session — activate the stack with the loader**:
```bash
source ~/load_dimon_env.sh
```
That single line:
- `module purge`
- `module load pytorch/2.6.0-py3-cu11.8` (which itself pulls in `python/3.13.3-gcc11`, `craype-accel-nvidia80`, `cuda/11.8.0`)
- exposes torch (2.6.0+cu118), numpy, scipy, matplotlib
- exposes the user-installed scikit-learn

The loader is the analogue of `~/load_opencarp_env.sh` (which activates the simulation stack). Use them in separate sessions, never together.

**GPU note**: `torch.cuda.is_available()` returns `False` on login nodes — that is expected. Submit GPU jobs to the `gpu` / `gpu1` / `gpu2` (production) or `gdev` (interactive dev) queues to actually train. Example interactive shell (replace walltime/resources for your queue's limits):
```bash
qsub -I -q gdev -l select=1:ncpus=8:ngpus=1:mem=64G -l walltime=01:00:00 -P personal-e1590340
# inside the interactive job:
source ~/load_dimon_env.sh
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**Do not** also activate `BASE/dimon/`. That venv is cray-python 3.9 and is reserved for PDF tooling (`pypdf`, `pdfplumber`); it has no torch and would mask the module's Python 3.13.

## Examples

This repository contains three examples: solving the Laplace equation on 2D domains, solving reaction-diffusion equations on 2D annulus, and predicting electrical wave propagation on patient-specific left ventricles.

1. Download data to the main folder (only required for `Laplace/`, `ReactionDiffusion/`, and the upstream `LV/` example with public data — patient-specific cardiac data is not redistributable).

2. Activate the environment:
   ```bash
   source ~/load_dimon_env.sh
   ```

3. Run training from the example folder:
   ```bash
   cd Laplace        # or ReactionDiffusion / LV / DIMON_PINN / etc.
   python main.py --epoch 10000
   ```

**Note**: please train with the same number of epochs to reproduce the results in the paper.

**Note**: the ten Tusscher model is adopted from the cellular model in CellML repository. ![tentusscher_2006](https://github.com/user-attachments/assets/1afb9b0a-e972-4876-8066-a0a774110e2e)

## Project context

This fork lives at `BASE/DIMON/` (where `BASE = ~/cardiac_simulation/`). It is the neural-operator side of the 12-lead ECG surrogate project — see `BASE/CLAUDE.md` for the full project overview, and `BASE/literature/dimon_notes.md` for our reading notes on the DIMON paper and how its example 3 maps onto our cohort.
