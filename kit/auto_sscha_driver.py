#!/usr/bin/env python3
"""Portable multi-stage SSCHA driver for shared /public/apps/sscha_kit deployments.

Features:
  - Relocatable kit + layered `profiles.yaml` lookup + `profile: "auto"` + `queue:`
  - User-controllable node budget (`max_nodes`, `nodes_per_job`) + optional walltime
  - Dynamic `worker_pool` mode (zero polling gap / zero wave barrier) + `batch` fallback
  - Full SSCHA_Minimizer control globally & per-stage:
      root_representation, preconditioning, min_step_dyn (lambda_a),
      min_step_struc (lambda_w), minim_struct (minim_struc), minim_dyn,
      kong_liu_ratio / n_random_eff, meaningful_factor, gradi_op, max_ka,
      abs_conv_thr, fixed_step, neglect_symmetries, use_spglib, enforce_sum_rule
  - Complete 4-level crash/power-outage recovery (`resume_from_stage: "auto"`):
      1) Stage-level (<stage>.DONE JSON)
      2) Population-level (dyn_pop<K>_ + ensembles_<stage>/)
      3) Mid-generation config-level (_active_generation/ + ESP_*.pwo JOB DONE)
      4) Post-relaxation Hessian/Jackknife/Plot level (zero extra DFT)

Usage: python auto_sscha_driver.py [job.yaml]
"""

import getpass
import json
import os
import re
import sys

import numpy as np
import yaml
import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.calculators
import sscha
import sscha.Ensemble
import sscha.SchaMinimizer
import sscha.Relax
import sscha.Utilities
from ase.data import atomic_masses, atomic_numbers

from worker_cluster import PortableCluster
from postprocess_sscha import (
    diag_irred_freqs_cm1,
    get_irred_q_indices,
    run_all_postprocess,
)

KIT_DIR = os.path.dirname(os.path.abspath(__file__))
EV_A3_TO_GPA = 160.2176634


def log(msg):
    print(msg, flush=True)


def load_profiles():
    """Search profiles.yaml in project -> ~/.config/sscha_kit -> KIT_DIR."""
    candidates = [
        os.path.abspath("profiles.yaml"),
        os.path.expanduser("~/.config/sscha_kit/profiles.yaml"),
        os.path.join(KIT_DIR, "profiles.yaml"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data, path
    raise FileNotFoundError("No profiles.yaml found in: %s" % ", ".join(candidates))


def select_profile(job, profiles, prof_path):
    """Resolve profile cleanly, supporting `profile: auto` and optional `queue:`."""
    req_prof = str(job.get("profile") or "auto").strip()
    req_queue = job.get("queue")

    if req_prof.lower() != "auto":
        if req_prof in profiles:
            return req_prof, profiles[req_prof]
        raise KeyError("Profile '%s' not found in %s (use profile: auto for cross-cluster portability)" % (req_prof, prof_path))

    if req_queue:
        for pname, pcfg in profiles.items():
            if str(pcfg.get("queue", "")).strip() == str(req_queue).strip():
                return pname, pcfg
        raise KeyError("Queue '%s' not found in any profile of %s" % (req_queue, prof_path))

    if "default" in profiles:
        return "default", profiles["default"]
    first_name = next(iter(profiles))
    return first_name, profiles[first_name]


def build_cluster(job, prof, job_name, stage_cfg=None):
    st = stage_cfg or {}
    hpc = PortableCluster(hostname="localhost")
    hpc.scheduler = prof.get("scheduler", "pbs").lower()
    hpc.submit_command = "qsub" if hpc.scheduler == "pbs" else "sbatch"
    hpc.submit_name = hpc.scheduler.upper()

    current_user = getpass.getuser()
    hpc.workdir = os.path.abspath("./_sscha_scratch_%s_%s" % (current_user, job_name))
    hpc.label = re.sub(r"[^A-Za-z0-9_]", "_", job_name)[:15] or "sscha_job"

    hpc.partition_name = prof["queue"]
    hpc.use_partition = True
    hpc.use_account = False
    hpc.use_memory = False
    hpc.active_stage_name = st.get("name", "stage1")
    hpc.n_nodes = int(st.get("nodes_per_job", job.get("nodes_per_job", prof.get("n_nodes", 1))))
    hpc.n_cpu = int(prof["n_cpu"])
    hpc.n_pool = int(st.get("n_pool", job.get("n_pool", prof.get("n_pool", 1))))
    hpc.batch_size = int(prof.get("batch_size", 4))
    hpc.max_nodes = st.get("max_nodes", job.get("max_nodes", None))
    if hpc.max_nodes is not None:
        hpc.max_nodes = int(hpc.max_nodes)
        hpc.batch_size = hpc.effective_n_workers()
    hpc.execution_mode = str(job.get("execution_mode", "worker_pool")).lower()
    hpc.job_number = int(prof.get("job_number", 1))
    hpc.max_recalc = int(job.get("max_recalc", 20))
    walltime = job.get("walltime", None)
    worker_walltime = job.get("worker_walltime", walltime)
    hpc.time = str(walltime) if walltime else None
    hpc.worker_walltime = str(worker_walltime) if worker_walltime else None
    hpc.use_time = bool(hpc.time or hpc.worker_walltime)
    hpc.check_timeout = job.get("check_timeout", 30)
    hpc.mpi_cmd = prof.get("mpi_cmd", "mpirun -hostfile $PBS_NODEFILE -np NPROC" if hpc.scheduler == "pbs" else "srun --mpi=pmi2 -n NPROC")
    hpc.binary = prof.get("binary", "pw.x -npool NPOOL -i PREFIX.pwi > PREFIX.pwo")
    hpc.load_modules = (
        prof["env_setup"].strip()
        + "\nulimit -s unlimited\nexport OMP_NUM_THREADS=1\n")
    return hpc


_MINIMIZER_KEYS = (
    "root_representation",
    "preconditioning",
    "min_step_dyn", "lambda_a",
    "min_step_struc", "lambda_w",
    "minim_struct", "minim_struc",
    "minim_dyn",
    "kong_liu_ratio", "n_random_eff",
    "meaningful_factor",
    "abs_conv_thr",
    "gradi_op",
    "max_ka",
    "fixed_step",
    "neglect_symmetries",
    "use_spglib",
    "enforce_sum_rule",
    "fix_cell_shape",
)


def resolve_stages(job):
    raw_stages = job.get("stages")
    target_press = job.get("target_press_gpa", None)

    if raw_stages:
        resolved = []
        for idx, st in enumerate(raw_stages, start=1):
            merged = dict(st)
            merged["index"] = idx
            merged["name"] = re.sub(r"[^A-Za-z0-9_]", "_", str(st.get("name", f"stage{idx}")))
            merged["n_configs"] = int(st.get("n_configs", job.get("n_configs", 100)))
            merged["max_pop"] = int(st.get("max_pop", job.get("max_pop", 10)))
            kpts_val = st["supercell_kpts"] if "supercell_kpts" in st else job["supercell_kpts"]
            merged["supercell_kpts"] = [int(x) for x in kpts_val]
            merged["conv_thr"] = float(st.get("conv_thr", job.get("conv_thr", 1.0e-8)))
            merged["ecutwfc"] = st["ecutwfc"] if "ecutwfc" in st else job["ecutwfc"]
            merged["ecutrho"] = st.get("ecutrho", job.get("ecutrho", merged["ecutwfc"] * 10))
            merged["degauss"] = float(st.get("degauss", job.get("degauss", 0.02)))
            merged["vc_relax"] = bool(st.get("vc_relax", target_press is not None))
            merged["fix_volume"] = bool(st.get("fix_volume", job.get("fix_volume", False)))
            merged["bulk_modulus"] = float(st.get("bulk_modulus", job.get("bulk_modulus", 300.0)))
            for k in _MINIMIZER_KEYS:
                if k not in merged and k in job:
                    merged[k] = job[k]
            resolved.append(merged)
        return resolved

    single = {
        "index": 1,
        "name": "stage1",
        "n_configs": int(job.get("n_configs", 100)),
        "max_pop": int(job.get("max_pop", 10)),
        "supercell_kpts": [int(x) for x in job["supercell_kpts"]],
        "conv_thr": float(job.get("conv_thr", 1.0e-8)),
        "ecutwfc": job["ecutwfc"],
        "ecutrho": job.get("ecutrho", job["ecutwfc"] * 10),
        "degauss": float(job.get("degauss", 0.02)),
        "vc_relax": target_press is not None,
        "fix_volume": bool(job.get("fix_volume", False)),
        "bulk_modulus": float(job.get("bulk_modulus", 300.0)),
    }
    for k in _MINIMIZER_KEYS:
        if k in job:
            single[k] = job[k]
    return [single]


def apply_minimizer_settings(minim, st):
    """Apply all user-configured SSCHA_Minimizer / sscha.x parameters cleanly."""
    if st.get("root_representation") is not None:
        minim.root_representation = str(st["root_representation"])
    if st.get("preconditioning") is not None:
        pflag = bool(st["preconditioning"])
        minim.precond_dyn = pflag
        minim.precond_wyck = pflag

    step_dyn = st.get("min_step_dyn", st.get("lambda_a"))
    minim.min_step_dyn = float(step_dyn if step_dyn is not None else 0.05)

    step_struc = st.get("min_step_struc", st.get("lambda_w"))
    if step_struc is not None:
        minim.min_step_struc = float(step_struc)

    m_struc = st.get("minim_struct", st.get("minim_struc"))
    if m_struc is not None:
        minim.minim_struct = bool(m_struc)
    if st.get("minim_dyn") is not None:
        minim.minim_dyn = bool(st["minim_dyn"])

    if st.get("n_random_eff") is not None and st.get("kong_liu_ratio") is None:
        minim.kong_liu_ratio = float(st["n_random_eff"]) / float(st["n_configs"])
    else:
        minim.kong_liu_ratio = float(st.get("kong_liu_ratio", 0.5))

    if st.get("meaningful_factor") is not None:
        minim.meaningful_factor = float(st["meaningful_factor"])
    if st.get("abs_conv_thr") is not None:
        minim.abs_conv_thr = float(st["abs_conv_thr"])
    if st.get("gradi_op") is not None:
        minim.gradi_op = str(st["gradi_op"])
    if st.get("max_ka") is not None:
        minim.max_ka = int(st["max_ka"])
    if st.get("fixed_step") is not None:
        minim.fixed_step = bool(st["fixed_step"])
    if st.get("neglect_symmetries") is not None:
        minim.neglect_symmetries = bool(st["neglect_symmetries"])
    if st.get("use_spglib") is not None:
        minim.use_spglib = bool(st["use_spglib"])
    if st.get("enforce_sum_rule") is not None:
        minim.enforce_sum_rule = bool(st["enforce_sum_rule"])


def build_qe_input_params(job, stage_cfg):
    pseudo_dir = job.get("pseudo_dir", os.path.join(KIT_DIR, "pseudos"))
    input_params = {
        "control": {
            "calculation": "scf",
            "tprnfor": True,
            "tstress": True,
            "disk_io": "none",
            "pseudo_dir": pseudo_dir,
        },
        "system": {
            "ecutwfc": stage_cfg["ecutwfc"],
            "ecutrho": stage_cfg["ecutrho"],
            "occupations": job.get("occupations", "smearing"),
            "smearing": job.get("smearing", "mp"),
            "degauss": stage_cfg["degauss"],
            "nosym": stage_cfg.get("nosym", job.get("nosym", True)),
            "noinv": stage_cfg.get("noinv", job.get("noinv", True)),
        },
        "electrons": {
            "conv_thr": float(stage_cfg["conv_thr"]),
        },
    }
    if job.get("input_dft"):
        input_params["system"]["input_dft"] = job["input_dft"]
    if job.get("mixing_beta") is not None:
        input_params["electrons"]["mixing_beta"] = float(stage_cfg.get("mixing_beta", job["mixing_beta"]))
    if job.get("electron_maxstep") is not None:
        input_params["electrons"]["electron_maxstep"] = int(stage_cfg.get("electron_maxstep", job["electron_maxstep"]))
    return input_params


def build_qe_calculator(job, prof, stage_cfg):
    kpts = tuple(int(x) for x in stage_cfg["supercell_kpts"])
    input_params = build_qe_input_params(job, stage_cfg)
    pseudos = dict(job["pseudos"])
    masses = {}
    for sym in pseudos:
        z = atomic_numbers.get(sym)
        masses[sym] = round(float(atomic_masses[z]), 4) if z else 1.0
    pdir = input_params["control"]["pseudo_dir"]
    for sym, fname in pseudos.items():
        fpath = os.path.join(pdir, fname)
        if not os.path.isfile(fpath):
            raise FileNotFoundError("Pseudopotential for %s not found: %s" % (sym, fpath))
    calc = CC.calculators.Espresso(
        input_data=input_params,
        pseudopotentials=pseudos,
        masses=masses,
        command="%s -in PREFIX.pwi > PREFIX.pwo" % prof.get("binary", "pw.x"),
        kpts=kpts,
        koffset=(0, 0, 0),
    )
    return calc, kpts, input_params


def compute_hessian_with_jackknife(
    ens,
    include_v4=False,
    use_symmetries=True,
    n_blocks=10,
    irred_idx=None,
):
    log("Computing full-ensemble free-energy Hessian (N=%d, include_v4=%s, use_symmetries=%s)..."
        % (ens.N, include_v4, use_symmetries))
    dyn_hess = ens.get_free_energy_hessian(
        include_v4=include_v4,
        use_symmetries=use_symmetries,
    )
    if irred_idx is None:
        irred_idx = get_irred_q_indices(dyn_hess)

    hess_freqs = diag_irred_freqs_cm1(dyn_hess, irred_idx)
    hess_se = np.full_like(hess_freqs, np.nan)

    n_blocks = int(n_blocks)
    if n_blocks >= 2 and ens.N >= n_blocks * 5:
        log("Estimating Hessian frequency error bars via %d-block delete-block Jackknife (N=%d)..."
            % (n_blocks, ens.N))
        rng = np.random.default_rng(20261125)
        perm = rng.permutation(ens.N)
        block_ids = np.empty(ens.N, dtype=int)
        for b, chunk in enumerate(np.array_split(perm, n_blocks)):
            block_ids[chunk] = b

        jk_freqs = []
        for b in range(n_blocks):
            mask = (block_ids != b)
            sub_ens = ens.split(mask)
            h_sub = sub_ens.get_free_energy_hessian(
                include_v4=include_v4,
                use_symmetries=use_symmetries,
            )
            jk_freqs.append(diag_irred_freqs_cm1(h_sub, irred_idx))
            log("  Jackknife block %d/%d done (sub_N=%d)" % (b + 1, n_blocks, sub_ens.N))

        jk_arr = np.array(jk_freqs)
        jk_mean = np.mean(jk_arr, axis=0)
        hess_se = np.sqrt(((n_blocks - 1.0) / n_blocks) * np.sum((jk_arr - jk_mean) ** 2, axis=0))
        log("Jackknife error estimation complete (max SE = %.2f cm-1)." % float(np.nanmax(hess_se)))
    else:
        log("Skipping Jackknife error bars (n_blocks=%d, ens.N=%d)." % (n_blocks, ens.N))

    return dyn_hess, irred_idx, hess_freqs, hess_se


def extract_pressure_gpa(ens):
    try:
        if not getattr(ens, "has_stress", False):
            return None, None
        stress, stress_err = ens.get_stress_tensor()
        p_eva3 = float(np.trace(stress) / 3.0)
        p_err_eva3 = float(np.sqrt(np.sum(np.diag(stress_err) ** 2)) / 3.0)
        return p_eva3 * EV_A3_TO_GPA, p_err_eva3 * EV_A3_TO_GPA
    except Exception as exc:
        log("Warning: could not compute final pressure from stress tensor: %s" % exc)
        return None, None


def _find_latest_pop_in_dir(ens_dir, nqirr):
    """Return (pop_id, dyn) for highest completed population in ens_dir, or (0, None)."""
    if not os.path.isdir(ens_dir):
        return 0, None
    best_pop = 0
    for fn in os.listdir(ens_dir):
        m = re.match(r"^energies_supercell_population(\d+)\.dat$", fn)
        if m:
            pnum = int(m.group(1))
            if pnum > best_pop and os.path.exists(f"dyn_pop{pnum}_1"):
                best_pop = pnum
    if best_pop > 0:
        d = CC.Phonons.Phonons(f"dyn_pop{best_pop}_", nqirr=nqirr)
        d.Symmetrize()
        d.ForcePositiveDefinite()
        return best_pop, d
    return 0, None


def _read_done_sentinel(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read().strip()
        if txt.startswith("{"):
            return json.loads(txt)
    except Exception:
        pass
    return {}


def main(job_yaml_path):
    with open(job_yaml_path, "r", encoding="utf-8") as f:
        job = yaml.safe_load(f)
    profiles, prof_path = load_profiles()
    prof_name, prof = select_profile(job, profiles, prof_path)

    stages = resolve_stages(job)
    job_name = job.get("job_name", "sscha_job")
    hpc0 = build_cluster(job, prof, job_name, stages[0])
    log("SSCHA driver | user=%s profile=%s (%s) queue=%s mode=%s workers=%d nodes/job=%d ppn=%d stages=%d"
        % (getpass.getuser(), prof_name, prof_path, prof["queue"], hpc0.execution_mode,
           hpc0.effective_n_workers(), hpc0.n_nodes, hpc0.n_cpu, len(stages)))

    hpc0.setup_workdir()
    if not hpc0.CheckCommunication():
        raise RuntimeError(
            "ssh localhost failed; configure passwordless localhost login "
            "for this account first")

    T0 = float(job["temperature"])
    nqirr = int(job["nqirr"])
    out_prefix = job.get("output_dyn_prefix", "final_anharmonic_dyn_")
    target_press = job.get("target_press_gpa", None)
    raw_resume = job.get("resume_from_stage", "auto")
    auto_resume = str(raw_resume).lower() == "auto"
    resume_from_stage = 0 if auto_resume else int(raw_resume)

    dyn_harm = CC.Phonons.Phonons(job["init_dyn_prefix"], nqirr=nqirr)
    dyn_harm.Symmetrize()
    dyn_harm.ForcePositiveDefinite()

    current_dyn = dyn_harm.Copy()
    next_pop = 1
    stage_dyns = []
    stage_evo_prefixes = []
    stage_ledger = []
    final_ens = None
    final_dyn = None
    last_ens_dir = None
    last_stage_vc = False

    for st in stages:
        idx = st["index"]
        sname = st["name"]
        ckpt_prefix = f"{sname}_dyn_"
        done_sentinel = f"{sname}.DONE"
        evo_prefix = f"sscha_evolution_{sname}" if len(stages) > 1 else "sscha_evolution"
        ens_dir = f"ensembles_{sname}" if len(stages) > 1 else "ensembles_data"
        stage_evo_prefixes.append((sname, evo_prefix))
        last_ens_dir = ens_dir

        should_skip = (
            (auto_resume and os.path.isfile(done_sentinel) and os.path.exists(ckpt_prefix + "1"))
            or (not auto_resume and resume_from_stage > idx and os.path.exists(ckpt_prefix + "1"))
        )
        if should_skip:
            smeta = _read_done_sentinel(done_sentinel)
            log("=== STAGE %d/%d (%s) SKIPPED (auto-resuming from completed %s*) ==="
                % (idx, len(stages), sname, ckpt_prefix))
            current_dyn = CC.Phonons.Phonons(ckpt_prefix, nqirr=nqirr)
            current_dyn.Symmetrize()
            current_dyn.ForcePositiveDefinite()
            final_dyn = current_dyn.Copy()
            stage_dyns.append((sname, current_dyn.Copy()))
            stage_ledger.append({
                "index": idx, "name": sname,
                "n_configs": smeta.get("n_configs", st["n_configs"]),
                "max_pop": st["max_pop"], "kpts": st["supercell_kpts"],
                "conv_thr": st["conv_thr"], "vc_relax": st["vc_relax"],
                "converged": smeta.get("converged", "resumed"),
                "neff": smeta.get("neff"), "kl": smeta.get("kl"),
            })
            next_pop = int(smeta.get("next_pop", next_pop + st["max_pop"]))
            last_stage_vc = bool(st["vc_relax"] and (target_press is not None))
            continue

        hpc = build_cluster(job, prof, job_name, st)
        hpc.setup_workdir()
        qe_calc, kpts, _ = build_qe_calculator(job, prof, st)
        use_vc = bool(st["vc_relax"] and (target_press is not None))

        stage_start_pop = next_pop
        dyn_in = current_dyn.Copy()
        if auto_resume or resume_from_stage == idx:
            latest_pop, latest_dyn = _find_latest_pop_in_dir(ens_dir, nqirr)
            if latest_pop > 0 and latest_dyn is not None:
                log("=== RESUME: Found completed pop %d in %s (continuing from pop %d) ==="
                    % (latest_pop, ens_dir, latest_pop + 1))
                dyn_in = latest_dyn
                stage_start_pop = latest_pop + 1

        dyn_in.Symmetrize()
        dyn_in.ForcePositiveDefinite()

        ens = sscha.Ensemble.Ensemble(dyn_in, T0=T0, supercell=dyn_in.GetSupercell())
        restart_from_ens = False
        if hpc.execution_mode == "worker_pool":
            gen_meta = hpc.check_resumable_generation(sname, st["n_configs"])
            if gen_meta is not None:
                log("=== RESUME: Found interrupted generation in %s/_active_generation ===" % hpc.workdir)
                ens.load_bin(hpc._active_gen_dir(), population_id=0)
                ens.dyn_0 = dyn_in.Copy()
                ens.update_weights(dyn_in, T0)
                hpc.resume_active_generation(
                    ens, qe_calc, get_stress=True, timeout=hpc.check_timeout
                )
                os.makedirs(ens_dir, exist_ok=True)
                ens.save_bin(ens_dir, stage_start_pop)
                restart_from_ens = True

        remaining_pops = max(1, (next_pop + st["max_pop"] - 1) - stage_start_pop + 1)
        minim = sscha.SchaMinimizer.SSCHA_Minimizer(ens)
        apply_minimizer_settings(minim, st)

        log("=== STAGE %d/%d (%s): N=%d, max_pop=%d (start_pop=%d), kpts=%s, conv_thr=%g, "
            "vc=%s, minim_struct=%s, root=%s, meaningful=%.3g, KL=%.2f, max_ka=%d ==="
            % (idx, len(stages), sname, st["n_configs"], st["max_pop"], stage_start_pop,
               list(kpts), st["conv_thr"], use_vc, minim.minim_struct,
               minim.root_representation, minim.meaningful_factor,
               minim.kong_liu_ratio, minim.max_ka))

        os.makedirs(ens_dir, exist_ok=True)
        relax = sscha.Relax.SSCHA(
            minim,
            ase_calculator=qe_calc,
            N_configs=st["n_configs"],
            max_pop=stage_start_pop + remaining_pops - 1,
            save_ensemble=True,
            data_dir=ens_dir,
            cluster=hpc,
        )
        if st.get("fix_cell_shape") is not None:
            relax.fix_cell_shape = bool(st["fix_cell_shape"])
        io_info = sscha.Utilities.IOInfo()
        io_info.SetupSaving(evo_prefix)
        relax.setup_custom_functions(custom_function_post=io_info.CFP_SaveAll)

        if use_vc:
            conv = relax.vc_relax(
                target_press=float(target_press),
                static_bulk_modulus=st["bulk_modulus"],
                fix_volume=st["fix_volume"],
                start_pop=stage_start_pop,
                restart_from_ens=restart_from_ens,
            )
        else:
            conv = relax.relax(
                get_stress=True,
                start_pop=stage_start_pop,
                restart_from_ens=restart_from_ens,
            )

        next_pop = int(getattr(relax, "start_pop", stage_start_pop + remaining_pops))
        current_dyn = relax.minimizer.dyn.Copy()
        final_dyn = relax.minimizer.dyn
        final_ens = relax.minimizer.ensemble
        last_stage_vc = use_vc

        if not use_vc:
            final_ens.update_weights(final_dyn, T0)

        neff = float(final_ens.get_effective_sample_size())
        kl = neff / float(final_ens.N)
        current_dyn.save_qe(ckpt_prefix)
        with open(done_sentinel, "w", encoding="utf-8") as df:
            json.dump({
                "stage": sname, "converged": bool(conv), "next_pop": next_pop,
                "n_configs": int(final_ens.N), "neff": neff, "kl": kl,
            }, df, indent=2)
        stage_dyns.append((sname, current_dyn.Copy()))
        stage_ledger.append({
            "index": idx, "name": sname, "n_configs": int(final_ens.N),
            "max_pop": st["max_pop"], "kpts": list(kpts),
            "conv_thr": st["conv_thr"], "vc_relax": use_vc,
            "converged": bool(conv), "neff": neff, "kl": kl,
        })
        log("STAGE %d/%d (%s) done | converged=%s | N=%d, N_eff=%.1f (KL=%.3f) | ckpt=%s*"
            % (idx, len(stages), sname, conv, final_ens.N, neff, kl, ckpt_prefix))

    assert final_dyn is not None
    # If all stages were skipped via .DONE, reload final_ens from last_ens_dir for Hessian/pressure!
    if final_ens is None and last_ens_dir is not None:
        lp, _ = _find_latest_pop_in_dir(last_ens_dir, nqirr)
        if lp > 0:
            log("=== RESUME: Reloading final ensemble (pop %d) from %s for Hessian/postprocess ===" % (lp, last_ens_dir))
            final_ens = sscha.Ensemble.Ensemble(final_dyn, T0=T0, supercell=final_dyn.GetSupercell())
            final_ens.load_bin(last_ens_dir, population_id=lp)
            final_ens.dyn_0 = final_dyn.Copy()
            final_ens.update_weights(final_dyn, T0)

    final_dyn.save_qe(out_prefix)
    log("All stages finished. Final auxiliary dynamical matrix prefix: %s" % out_prefix)

    p_gpa, p_err = extract_pressure_gpa(final_ens) if final_ens is not None else (None, None)
    meta = {
        "stages": stage_ledger,
        "p_final_gpa": p_gpa,
        "p_final_err_gpa": p_err,
        "last_stage_vc": last_stage_vc,
    }

    dyn_hess = None
    hess_freqs = None
    hess_se = None
    irred_idx = get_irred_q_indices(final_dyn)

    if job.get("compute_hessian", True) and final_ens is not None:
        n_blocks = int(job.get("hessian_error_blocks", 10))
        include_v4 = bool(job.get("hessian_include_v4", False))
        dyn_hess, irred_idx, hess_freqs, hess_se = compute_hessian_with_jackknife(
            final_ens,
            include_v4=include_v4,
            use_symmetries=True,
            n_blocks=n_blocks,
            irred_idx=irred_idx,
        )
        dyn_hess.save_qe("hessian_" + out_prefix)
        log("Hessian saved with prefix: hessian_%s" % out_prefix)

    run_all_postprocess(
        job=job,
        dyn_harm=dyn_harm,
        stage_dyns=stage_dyns,
        dyn_final=final_dyn,
        dyn_hess=dyn_hess,
        irred_idx=irred_idx,
        hess_freqs=hess_freqs,
        hess_se=hess_se,
        meta=meta,
        stage_evo_prefixes=stage_evo_prefixes,
        out_dir=".",
    )


if __name__ == "__main__":
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "job.yaml"
    main(cfg_file)
