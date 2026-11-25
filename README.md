# sscha_kit

**Portable, Multi-Stage Automated SSCHA Workflow for PBS/Torque & SLURM Clusters**

[English](README.md) | [简体中文](README_CN.md)

> **Copyright & Attribution**: Copyright © 2026 **Defang Duan Research Group, Jilin University (吉林大学段德芳课题组)**. Independently developed by **Haoqi Chen (陈浩琦)**. Completely open-source and free to use under the [MIT License](LICENSE).

`sscha_kit` wraps [`python-sscha`](https://github.com/SSCHAcode/python-sscha) and [`CellConstructor`](https://github.com/SSCHAcode/CellConstructor) with Quantum ESPRESSO (`pw.x`) into a single-command, cluster-portable workflow featuring:

1. **Dynamic Worker-Pool Scheduling (`execution_mode: "worker_pool"`)**:
   - Submits only `max_nodes // nodes_per_job` long-lived worker jobs per generation instead of 1 job per configuration.
   - Workers claim configurations via atomic POSIX `mkdir` locks in a shared directory — **zero inter-configuration polling gaps, zero wave barriers, automatic load balancing**, and full **multi-node MPI (`nodes_per_job >= 2`)** support.
2. **Coarse-to-Dense Multi-Stage Schedules (`stages:`)**:
   - Seamlessly progress from coarse $k$-meshes & small ensembles (rapidly hardening soft modes and relaxing the cell) to dense $k$-meshes & large ensembles at fixed converged cell for final Free-Energy Hessian + **10-block Jackknife statistical error bars** (zero extra DFT cost).
   - Full `SSCHA_Minimizer` parameter exposure (`root_representation`, `preconditioning`, `min_step_dyn`/`lambda_a`, `min_step_struc`/`lambda_w`, `minim_struct`, `meaningful_factor`, `gradi_op`, `max_ka`, `kong_liu_ratio`/`n_random_eff`).
3. **4-Level Crash & Power-Outage Recovery (`resume_from_stage: "auto"`)**:
   - **Stage level**: skips completed stages (`<stage>.DONE`).
   - **Population level**: resumes from latest `dyn_pop<K>_` in the active stage.
   - **Mid-generation config level**: if power fails at 80/100 configs, verifies `JOB DONE` in existing `.pwo` files and **runs only the remaining 20 configs** on the exact saved ensemble.
   - **Post-relaxation level**: resumes Hessian/Jackknife/plotting without re-running DFT.
4. **Cross-Cluster Portability & Shared `/public` Deployment**:
   - `scripts/init_server.sh` auto-detects PBS/SLURM, `pw.x`, queues, and core counts to generate `profiles.yaml` with a `default` entry (`profile: "auto"` in `job.yaml` requires zero edits across clusters).
   - Automatic `PYTHONPATH` / Conda environment isolation across multi-user shared installations.
5. **Automatic Post-Processing**:
   - Generates `hessian_frequencies_summary.csv`, `sscha_final_summary.txt`, `sscha_phonon_comparison.png` (300 DPI), and `sscha_convergence_evolution.png` (300 DPI).

---

## Installation

### Option A: Shared Cluster Deployment (from offline `conda-pack` tarball or online environment)

1. Place `sscha_env.tar.gz` (optional, for offline clusters) in the repo root, or install Python dependencies via pip/conda:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the shared installer (auto-detects queues & `pw.x`, sets `a+rX` permissions):
   ```bash
   bash scripts/install_public.sh /public/apps/sscha_kit
   ```
3. *(Optional)* Register global shortcuts so users can run `sscha-check` and `sscha-run` from any directory:
   ```bash
   ln -sf /public/apps/sscha_kit/check.sh /usr/local/bin/sscha-check
   ln -sf /public/apps/sscha_kit/run.sh   /usr/local/bin/sscha-run
   ```

### Option B: User-Space Installation
```bash
bash scripts/install_public.sh "$HOME/sscha_kit"
```
Ensure passwordless `ssh localhost` is enabled for the executing user (required by `sscha.Cluster` communication checks).

---

## Quick Start

Inside a directory containing `harmonic_dyn0..nqirr` and `./pseudo/*.UPF`:

1. **Copy and edit `job.yaml`**:
   ```bash
   cp /public/apps/sscha_kit/job.yaml.example ./job.yaml
   ```
   Key fields: `job_name`, `profile: "auto"` (or optional `queue: "<name>"`), `max_nodes`, `nodes_per_job`, `temperature`, `target_press_gpa`, and `pseudos`.
2. **Dry-run verification (submits nothing)**:
   ```bash
   sscha-check
   # or: bash /public/apps/sscha_kit/check.sh
   ```
   Wait for `ALL_CHECKS_PASSED (no job submitted)`.
3. **Launch**:
   - **Daemon mode** (login node):
     ```bash
     sscha-run
     # or: bash /public/apps/sscha_kit/run.sh
     ```
   - **Batch-hosted mode** (PBS / SLURM):
     ```bash
     # PBS
     qsub -q batch -l nodes=1:ppn=2 -v SSCHA_KIT_DIR=/public/apps/sscha_kit /public/apps/sscha_kit/run_in_job.sh
     # SLURM
     sbatch -p cpu -N 1 -n 2 --wrap="bash /public/apps/sscha_kit/run_in_job.sh job.yaml"
     ```

---

## How Lattice & Atomic Relaxation Work Together

When `vc_relax: true` is active in a stage, `python-sscha` automatically couples **atomic centroid relaxation (`minim_struct: true`)**, **force-constant optimization (`minim_dyn: true`)**, and **variable-cell relaxation** inside every population:
1. First, `SSCHA_Minimizer` relaxes both force constants (`min_step_dyn`) and atomic positions (`min_step_struc`) at the current cell.
2. Next, the anharmonic stress tensor updates the unit cell (`bulk_modulus`).
3. Both inner gradients and target pressure must converge simultaneously before the stage finishes.

A standard 3-stage workflow (`stage1_coarse [vc_relax: true]` $\to$ `stage2_medium/dense [vc_relax: true]` $\to$ `stage3_hessian [vc_relax: false, large N]`) locks the converged equilibrium cell in the final stage so the Free-Energy Hessian ($D^{(2)} + D^{(3)}$) is evaluated on a strictly consistent lattice.

---

## Output Files

| File | Description |
| :--- | :--- |
| `final_anharmonic_dyn_0..N` | Converged auxiliary SSCHA dynamical matrices (positive definite) |
| `hessian_final_anharmonic_dyn_0..N` | Physical Free-Energy Hessian dynamical matrices ($D^{(2)} + D^{(3)}$) |
| `hessian_frequencies_summary.csv` | Mode-resolved frequency table across all stages + Jackknife SE |
| `sscha_final_summary.txt` | Convergence ledger, lattice/volume changes, final pressure, stability verdict |
| `sscha_phonon_comparison.png` | 300-DPI figure: irreducible $q$ soft modes ($\pm$ SE) & interpolated dispersion |
| `sscha_convergence_evolution.png` | 300-DPI multi-stage free energy and gradient norm evolution |

---

## License

Copyright © 2026 **Defang Duan Research Group, Jilin University (吉林大学段德芳课题组)**.
Independently developed by **Haoqi Chen (陈浩琦)**.
Completely open-source and free to use under the [MIT License](LICENSE).
