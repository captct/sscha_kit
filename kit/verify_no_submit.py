#!/usr/bin/env python3
"""Kit verification WITHOUT submitting any job.

Checks: dyn load + symmetrize, local ensemble generation, ALL stages'
CellConstructor Espresso calculators, layered profiles.yaml resolution,
node-budget (`max_nodes` / `nodes_per_job`) + dynamic `worker_pool` script
dry-run (printed only, never passed to qsub/sbatch).

Usage: python verify_no_submit.py [project_dir]   (default: cwd)
"""

import os
import sys
import tempfile

import yaml
import cellconstructor as CC
import cellconstructor.Phonons
import sscha
import sscha.Ensemble
import sscha.SchaMinimizer

from auto_sscha_driver import (
    apply_minimizer_settings,
    build_cluster,
    build_qe_calculator,
    load_profiles,
    resolve_stages,
    select_profile,
)


def main():
    proj = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
    os.chdir(proj)

    with open("job.yaml", encoding="utf-8") as f:
        job = yaml.safe_load(f)
    profiles, prof_path = load_profiles()
    prof_name, prof = select_profile(job, profiles, prof_path)
    print("PROFILES_LOADED path=%s profile=%s (queue=%s, ppn=%s)" % (
        prof_path, prof_name, prof["queue"], prof["n_cpu"]))

    dyn = CC.Phonons.Phonons(job["init_dyn_prefix"], nqirr=job["nqirr"])
    dyn.Symmetrize()
    dyn.ForcePositiveDefinite()
    print("DYN_LOAD_OK atoms=%d supercell=%s"
          % (dyn.structure.N_atoms, dyn.GetSupercell()))

    stages = resolve_stages(job)
    ens = sscha.Ensemble.Ensemble(
        dyn, T0=float(job["temperature"]), supercell=dyn.GetSupercell())
    ens.generate(int(stages[0]["n_configs"]))
    print("ENSEMBLE_OK stage1_N=%d supercell_atoms=%d" % (ens.N, ens.structures[0].N_atoms))

    print("MULTI_STAGE_PLAN (%d stages, execution_mode=%s, jk_blocks=%d):"
          % (len(stages), job.get("execution_mode", "worker_pool"),
             int(job.get("hessian_error_blocks", 10))))
    with tempfile.TemporaryDirectory() as tmpd:
        for st in stages:
            calc, kpts, inp = build_qe_calculator(job, prof, st)
            hpc_st = build_cluster(job, prof, job.get("job_name", "verify"), st)
            c_copy = calc.copy()
            c_copy.set_directory(tmpd)
            lbl = "ESP_verify_s%d" % st["index"]
            c_copy.set_label(lbl)
            c_copy.write_input(ens.structures[0])
            pwi = os.path.join(tmpd, lbl + ".pwi")
            assert os.path.isfile(pwi), "Missing generated pwi: %s" % pwi
            print("  [Stage %d: %-15s] N=%-3d kpts=%-9s workers=%dx%dnodes (%d cores/job) -> PWI_OK"
                  % (st["index"], st["name"], st["n_configs"], list(kpts),
                     hpc_st.effective_n_workers(), hpc_st.n_nodes, hpc_st.total_nproc_per_job()))
            m_test = sscha.SchaMinimizer.SSCHA_Minimizer(ens)
            apply_minimizer_settings(m_test, st)
            print("    minimizer: vc=%s minim_struct=%s root=%s step_dyn=%.3g step_struc=%.3g meaningful=%.3g KL=%.2f gradi=%s max_ka=%d"
                  % (bool(st["vc_relax"] and job.get("target_press_gpa") is not None),
                     m_test.minim_struct, m_test.root_representation,
                     m_test.min_step_dyn, m_test.min_step_struc,
                     m_test.meaningful_factor, m_test.kong_liu_ratio,
                     m_test.gradi_op, m_test.max_ka))

    hpc = build_cluster(job, prof, job.get("job_name", "verify"), stages[0])
    hpc.setup_workdir()
    print("SSH_LOCALHOST_OK=%s" % hpc.CheckCommunication())

    if hpc.execution_mode == "worker_pool":
        script = hpc.create_worker_script(0, stages[0]["n_configs"], os.path.join(hpc.workdir, "_pool_locks_dryrun"))
        print("---- WORKER_POOL SCRIPT (worker 0 dry-run, NOT submitted) ----")
    else:
        script = hpc.create_submission_script(["ESP_0"])
        print("---- BATCH SCRIPT (dry-run, NOT submitted) ----")
    print(script)
    print("ALL_CHECKS_PASSED (no job submitted)")


if __name__ == "__main__":
    main()
