"""Unified Torque/PBS + SLURM cluster adapter with dynamic Worker-Pool mode
and mid-generation crash/power-outage recovery.

Two execution modes (configured via `execution_mode` in job.yaml):
  1. `worker_pool` (DEFAULT, recommended):
     Submits ONLY `n_workers` long-lived jobs (`max(1, max_nodes // nodes_per_job)`).
     Each worker runs an atomic POSIX `mkdir`-lock work-stealing loop over all N
     configurations: zero polling gap, zero wave barrier, dynamic load balancing,
     and full multi-node MPI (`nodes_per_job >= 1`).
     Crash / power-outage recovery:
       - Before launching workers, saves the current generation's ensemble to
         `<workdir>/_active_generation/` (`population0_*` + `meta.json`).
       - On restart after an outage (e.g. 80/100 finished), `resume_active_generation`
         scans `ESP_*.pwo` for `"JOB DONE"`, skips all 80 finished configs, clears
         stale locks, and launches workers ONLY for the remaining 20 configs!
  2. `batch` (fallback):
     Standard `sscha.Cluster.compute_ensemble_batch`.
"""

import json
import os
import shutil
import subprocess
import time

import ase.units as ase_units
import numpy as np
import sscha.Cluster


class PortableCluster(sscha.Cluster.Cluster):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__total_attributes__.extend([
            "scheduler", "execution_mode", "max_nodes",
            "nodes_per_job", "worker_walltime", "use_ssh_localhost",
            "active_stage_name",
        ])
        self.scheduler = "pbs"
        self.execution_mode = "worker_pool"
        self.max_nodes = None
        self.nodes_per_job = 1
        self.worker_walltime = None
        self.use_ssh_localhost = True
        self.active_stage_name = "stage1"

    def total_nproc_per_job(self):
        return int(self.n_nodes) * int(self.n_cpu)

    def effective_n_workers(self):
        if self.max_nodes is not None:
            return max(1, int(self.max_nodes) // max(1, int(self.n_nodes)))
        return max(1, int(self.batch_size))

    def _run_local_or_ssh(self, cmd):
        if self.use_ssh_localhost:
            return self.ExecuteCMD(cmd, False, return_output=True, on_cluster=True)
        proc = subprocess.run(cmd, shell=True, text=True, capture_output=True)
        return (proc.returncode == 0), (proc.stdout + proc.stderr).strip()

    def _build_header(self, job_label, walltime):
        nproc = self.total_nproc_per_job()
        s = "#!" + self.terminal + "\n"
        if self.scheduler == "pbs":
            s += "#PBS -N %s\n" % str(job_label)[:15]
            if self.use_partition and self.partition_name:
                s += "#PBS -q %s\n" % self.partition_name
            s += "#PBS -l nodes=%d:ppn=%d\n" % (self.n_nodes, self.n_cpu)
            if self.use_time and walltime:
                s += "#PBS -l walltime=%s\n" % walltime
            if self.use_memory and self.ram:
                s += "#PBS -l mem=%s\n" % self.ram
            s += "#PBS -j oe\n"
        elif self.scheduler == "slurm":
            s += "#SBATCH --job-name=%s\n" % str(job_label)[:15]
            if self.use_partition and self.partition_name:
                s += "#SBATCH --partition=%s\n" % self.partition_name
            s += "#SBATCH --nodes=%d\n" % self.n_nodes
            s += "#SBATCH --ntasks-per-node=%d\n" % self.n_cpu
            s += "#SBATCH --ntasks=%d\n" % nproc
            if self.use_time and walltime:
                s += "#SBATCH --time=%s\n" % walltime
            s += "#SBATCH --output=%s_%%j.out\n" % str(job_label)[:15]
        else:
            raise ValueError("Unsupported scheduler: %s" % self.scheduler)

        for key, value in self.custom_params.items():
            prefix = "#PBS" if self.scheduler == "pbs" else "#SBATCH"
            if value is None:
                s += "%s %s\n" % (prefix, key)
            else:
                s += "%s %s=%s\n" % (prefix, key, value)

        s += self.load_modules + "\n"
        return s

    def create_submission_script(self, labels):
        submission = self._build_header(self.label, self.time)
        if self.add_set_minus_x:
            submission += "set -x\n"
        submission += "cd " + self.workdir + "\n"
        other_input, other_output = ("", "")
        if self.additional_script_parameters is not None:
            other_input, other_output = self.additional_script_parameters(labels)
        submission += other_input
        nproc = self.total_nproc_per_job()
        new_mpicmd = self.mpi_cmd.replace("NPROC", str(nproc))
        for lbl in labels:
            binary = self.binary.replace("NPOOL", str(self.n_pool)).replace("PREFIX", lbl)
            submission += "%s %s\n" % (new_mpicmd, binary)
        submission += other_output
        return submission

    def create_worker_script(self, worker_id, n_configs, lock_dir):
        wlabel = "%s_w%d" % (str(self.label)[:11], worker_id)
        script = self._build_header(wlabel, self.worker_walltime)
        nproc = self.total_nproc_per_job()
        new_mpicmd = self.mpi_cmd.replace("NPROC", str(nproc))
        binary_tpl = self.binary.replace("NPOOL", str(self.n_pool)).replace("PREFIX", "${LBL}")

        script += f"""
cd "{self.workdir}"
LOCK_DIR="{lock_dir}"
mkdir -p "$LOCK_DIR"

echo "[$(date '+%F %T')] Worker {worker_id} started on $(hostname) (nproc={nproc}, npool={self.n_pool})"
while true; do
    CLAIMED=""
    for idx in $(seq 0 {n_configs - 1}); do
        if [ -f "$LOCK_DIR/done_${{idx}}" ] || [ -f "$LOCK_DIR/fail_${{idx}}" ]; then
            continue
        fi
        if mkdir "$LOCK_DIR/claim_${{idx}}" 2>/dev/null; then
            CLAIMED="$idx"
            break
        fi
    done

    if [ -z "$CLAIMED" ]; then
        echo "[$(date '+%F %T')] Worker {worker_id}: no remaining configs, exiting cleanly."
        break
    fi

    LBL="ESP_${{CLAIMED}}"
    echo "[$(date '+%F %T')] Worker {worker_id} claiming $LBL"
    {new_mpicmd} {binary_tpl}
    RC=$?
    if grep -q "JOB DONE" "${{LBL}}.pwo" 2>/dev/null; then
        touch "$LOCK_DIR/done_${{CLAIMED}}"
        echo "[$(date '+%F %T')] Worker {worker_id} finished $LBL (rc=$RC)"
    else
        echo "[$(date '+%F %T')] Worker {worker_id} FAILED $LBL (rc=$RC), marking fail_${{CLAIMED}} to prevent spin"
        touch "$LOCK_DIR/fail_${{CLAIMED}}"
    fi
done
"""
        return script

    def check_job_finished(self, job_id, verbose=True):
        if self.scheduler == "slurm":
            cmd = "squeue -h -j %s -o '%%T' 2>/dev/null; echo SSCHA_STAT_DONE" % job_id
            _, output = self._run_local_or_ssh(cmd)
            if "SSCHA_STAT_DONE" not in output:
                return False
            body = output.replace("SSCHA_STAT_DONE", "").strip()
            if not body:
                return True
            state = body.splitlines()[0].strip().upper()
            return state in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL")

        cmd = "qstat -f %s 2>/dev/null; echo SSCHA_QSTAT_DONE" % job_id
        _, output = self._run_local_or_ssh(cmd)
        if "SSCHA_QSTAT_DONE" not in output:
            return False
        body = output.replace("SSCHA_QSTAT_DONE", "").strip()
        if not body:
            return True
        for line in body.splitlines():
            if "job_state" in line:
                state = line.split("=")[-1].strip()
                return state in ("C", "E")
        return False

    def _submit_script_text(self, script_path):
        submit_cmd = "qsub" if self.scheduler == "pbs" else "sbatch"
        ok, out = self._run_local_or_ssh(f"{submit_cmd} {script_path}")
        if not ok:
            raise RuntimeError(f"Job submission failed ({submit_cmd} {script_path}): {out}")
        return out.strip().splitlines()[-1].split()[-1]

    def _active_gen_dir(self):
        return os.path.join(self.workdir, "_active_generation")

    def _has_valid_pwo(self, idx, cellconstructor_calc, get_stress):
        pwo = os.path.join(self.workdir, f"ESP_{idx}.pwo")
        if not os.path.isfile(pwo):
            return False
        try:
            with open(pwo, "r", encoding="utf-8", errors="ignore") as fp:
                if "JOB DONE" not in fp.read():
                    return False
            c = cellconstructor_calc.copy()
            c.set_directory(self.workdir)
            c.set_label(f"ESP_{idx}")
            c.read_results()
            res = c.results
            if res is None or "energy" not in res or "forces" not in res:
                return False
            if get_stress and "stress" not in res:
                return False
            return True
        except Exception:
            return False

    def _run_workers_and_parse(self, ensemble, cellconstructor_calc, lock_dir, get_stress=True, timeout=None):
        n_total = ensemble.N
        ensemble.has_stress = get_stress
        done_now = sum(1 for i in range(n_total) if os.path.exists(os.path.join(lock_dir, f"done_{i}")))
        remaining = n_total - done_now
        pop_tag = time.strftime("%Y%m%d_%H%M%S")

        if remaining > 0:
            n_workers = min(self.effective_n_workers(), remaining)
            print(
                f"[WORKER_POOL] Launching {n_workers} worker jobs for {remaining}/{n_total} remaining configs "
                f"(nodes_per_job={self.n_nodes}, ppn={self.n_cpu}, nproc/job={self.total_nproc_per_job()}, npool={self.n_pool})",
                flush=True,
            )

            def _launch_workers(count, tag):
                ids = []
                for w in range(count):
                    wscript_path = os.path.join(self.workdir, f"_worker_{tag}_{w}.sh")
                    with open(wscript_path, "w", encoding="utf-8", newline="\n") as fp:
                        fp.write(self.create_worker_script(w, n_total, lock_dir))
                    jid = self._submit_script_text(wscript_path)
                    print(f"[WORKER_POOL] Submitted worker {w}: job_id={jid}", flush=True)
                    ids.append(jid)
                return ids

            active_jobs = _launch_workers(n_workers, pop_tag)
            poll_sec = float(timeout if timeout is not None else self.check_timeout)
            round_idx = 0

            while True:
                time.sleep(poll_sec)
                done_count = sum(
                    1 for i in range(n_total)
                    if os.path.exists(os.path.join(lock_dir, f"done_{i}"))
                )
                still_running = [jid for jid in active_jobs if not self.check_job_finished(jid)]
                print(
                    f"[WORKER_POOL] Progress: {done_count}/{n_total} configs done | "
                    f"active_workers={len(still_running)}/{len(active_jobs)}",
                    flush=True,
                )
                if done_count >= n_total and not still_running:
                    break
                if not still_running and done_count < n_total:
                    round_idx += 1
                    if round_idx > int(self.max_recalc):
                        raise RuntimeError(
                            f"[WORKER_POOL] Exceeded max_recalc={self.max_recalc} with "
                            f"{n_total - done_count}/{n_total} configs still incomplete."
                        )
                    for i in range(n_total):
                        done_f = os.path.join(lock_dir, f"done_{i}")
                        claim_d = os.path.join(lock_dir, f"claim_{i}")
                        if not os.path.exists(done_f) and os.path.isdir(claim_d):
                            try:
                                os.rmdir(claim_d)
                            except OSError:
                                pass
                    rem = n_total - done_count
                    n_respawn = min(self.effective_n_workers(), rem)
                    print(f"[WORKER_POOL] Respawning {n_respawn} workers for {rem} remaining configs (round {round_idx})...", flush=True)
                    active_jobs = _launch_workers(n_respawn, f"{pop_tag}_r{round_idx}")
        else:
            print(f"[WORKER_POOL] All {n_total}/{n_total} configs already finished on disk!", flush=True)

        # Parse all N outputs (CC.calculators.Espresso returns eV, eV/A, eV/A^3)
        for i in range(n_total):
            lbl = f"ESP_{i}"
            c = cellconstructor_calc.copy()
            c.set_directory(self.workdir)
            c.set_label(lbl)
            c.read_results()
            res = c.results
            if res is None or "energy" not in res or "forces" not in res:
                raise RuntimeError(f"[WORKER_POOL] Missing energy/forces in {self.workdir}/{lbl}.pwo")
            if get_stress and "stress" not in res:
                raise RuntimeError(f"[WORKER_POOL] Missing stress in {self.workdir}/{lbl}.pwo")

            ensemble.energies[i] = res["energy"] / ase_units.Ry
            ensemble.forces[i, :, :] = res["forces"] / ase_units.Ry
            ensemble.force_computed[i] = True
            if get_stress:
                stress = np.zeros((3, 3), dtype=np.float64)
                stress[0, 0], stress[1, 1], stress[2, 2] = res["stress"][0], res["stress"][1], res["stress"][2]
                stress[1, 2] = stress[2, 1] = res["stress"][3]
                stress[0, 2] = stress[2, 0] = res["stress"][4]
                stress[0, 1] = stress[1, 0] = res["stress"][5]
                ensemble.stresses[i, :, :] = -stress * ase_units.Bohr ** 3 / ase_units.Ry
                ensemble.stress_computed[i] = True

        # Generation finished cleanly -> clear active generation checkpoint
        act_dir = self._active_gen_dir()
        if os.path.isdir(act_dir):
            shutil.rmtree(act_dir, ignore_errors=True)
        print(f"[WORKER_POOL] All {n_total}/{n_total} configurations parsed successfully.", flush=True)

    def check_resumable_generation(self, stage_name, n_configs):
        """Return active-generation metadata dict if an interrupted generation matches."""
        meta_path = os.path.join(self._active_gen_dir(), "meta.json")
        if not os.path.isfile(meta_path):
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("stage_name") == stage_name and int(meta.get("n_total", -1)) == int(n_configs):
                return meta
        except Exception:
            return None
        return None

    def resume_active_generation(self, ensemble, cellconstructor_calc, get_stress=True, timeout=None):
        """Recover an interrupted generation: keep valid finished `.pwo`, run only remaining."""
        act_dir = self._active_gen_dir()
        meta_path = os.path.join(act_dir, "meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        lock_dir = meta["lock_dir"]
        os.makedirs(lock_dir, exist_ok=True)
        n_total = ensemble.N

        # Reconcile actual valid `.pwo` files on disk vs lock_dir
        reused = 0
        for i in range(n_total):
            done_f = os.path.join(lock_dir, f"done_{i}")
            claim_d = os.path.join(lock_dir, f"claim_{i}")
            if self._has_valid_pwo(i, cellconstructor_calc, get_stress):
                if not os.path.exists(done_f):
                    open(done_f, "w").close()
                reused += 1
            else:
                if os.path.exists(done_f):
                    os.remove(done_f)
                if os.path.isdir(claim_d):
                    try:
                        os.rmdir(claim_d)
                    except OSError:
                        pass
                # Ensure `.pwi` exists (rewrite if missing/corrupt)
                pwi = os.path.join(self.workdir, f"ESP_{i}.pwi")
                if not os.path.isfile(pwi):
                    c = cellconstructor_calc.copy()
                    c.set_directory(self.workdir)
                    c.set_label(f"ESP_{i}")
                    c.write_input(ensemble.structures[i])

        print(
            f"[RESUME] Recovered interrupted generation ({meta.get('stage_name')}): "
            f"{reused}/{n_total} configs already valid on disk, running remaining {n_total - reused}!",
            flush=True,
        )
        self._run_workers_and_parse(ensemble, cellconstructor_calc, lock_dir, get_stress=get_stress, timeout=timeout)

    def compute_ensemble_worker_pool(self, ensemble, cellconstructor_calc, get_stress=True, timeout=None):
        n_total = ensemble.N
        os.makedirs(self.local_workdir, exist_ok=True)
        os.makedirs(self.workdir, exist_ok=True)

        pop_tag = time.strftime("%Y%m%d_%H%M%S")
        lock_dir = os.path.join(self.workdir, f"_pool_locks_{pop_tag}")
        os.makedirs(lock_dir, exist_ok=True)

        # 1. Persist active-generation ensemble BEFORE submitting workers (crash-recovery checkpoint)
        act_dir = self._active_gen_dir()
        if os.path.isdir(act_dir):
            shutil.rmtree(act_dir, ignore_errors=True)
        os.makedirs(act_dir, exist_ok=True)
        ensemble.save_bin(act_dir, population_id=0)
        with open(os.path.join(act_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({
                "stage_name": self.active_stage_name,
                "n_total": int(n_total),
                "lock_dir": lock_dir,
                "get_stress": bool(get_stress),
            }, f, indent=2)

        # 2. Write all N `.pwi` files
        same_fs = os.path.abspath(self.local_workdir) == os.path.abspath(self.workdir)
        target_dir = self.workdir if same_fs else self.local_workdir
        print(f"[WORKER_POOL] Preparing {n_total} QE inputs in {self.workdir}...", flush=True)
        for i in range(n_total):
            lbl = f"ESP_{i}"
            for old_f in (
                os.path.join(self.workdir, lbl + ".pwo"),
                os.path.join(self.local_workdir, lbl + ".pwo"),
            ):
                if os.path.exists(old_f):
                    os.remove(old_f)
            c = cellconstructor_calc.copy()
            c.set_directory(target_dir)
            c.set_label(lbl)
            c.write_input(ensemble.structures[i])
            if not same_fs:
                self.copy_file(os.path.join(self.local_workdir, lbl + ".pwi"), self.workdir)

        self._run_workers_and_parse(ensemble, cellconstructor_calc, lock_dir, get_stress=get_stress, timeout=timeout)

    def compute_ensemble(self, ensemble, ase_calc, get_stress=True, timeout=None):
        if str(self.execution_mode).lower() == "worker_pool":
            self.compute_ensemble_worker_pool(ensemble, ase_calc, get_stress, timeout)
        else:
            self.compute_ensemble_batch(ensemble, ase_calc, get_stress, timeout)
