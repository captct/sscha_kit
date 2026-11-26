#!/usr/bin/env python3
"""Automatic post-processing & 300-DPI figure generation for the multi-stage SSCHA kit.

Called automatically at the end of auto_sscha_driver.py (headless Agg backend),
and can also be executed standalone in any completed project directory:
    /path/to/sscha_kit/sscha_env/bin/python /path/to/sscha_kit/postprocess_sscha.py [project_dir]

Outputs:
  1. hessian_frequencies_summary.csv  : Per-q, per-mode frequency evolution across all stages + Hessian SE
  2. sscha_final_summary.txt          : Complete report (all stages, cell, pressure, soft modes, stability)
  3. sscha_phonon_comparison.png      : 300-DPI 2-panel figure (irred q-points with SE bars + dispersion)
  4. sscha_convergence_evolution.png  : 300-DPI multi-stage evolution (free energy & gradient norms)
"""

import csv
import os
import sys

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.ForceTensor
import cellconstructor.Methods


def get_irred_q_indices(dyn):
    """Return representative indices in dyn.q_tot for each irreducible q star."""
    if hasattr(dyn, "q_stars") and dyn.q_stars:
        irred_idx = []
        q_tot = np.array(dyn.q_tot)
        for star in dyn.q_stars:
            q0 = np.array(star[0])
            dists = np.linalg.norm(q_tot - q0, axis=1)
            irred_idx.append(int(np.argmin(dists)))
        return irred_idx
    return list(range(len(dyn.q_tot)))


def diag_irred_freqs_cm1(dyn, irred_idx):
    """Diagonalize dyn at each irreducible q-point -> array(nqirr, 3*nat) in cm-1."""
    rows = []
    for iq in irred_idx:
        w, _ = dyn.DyagDinQ(iq)
        rows.append(np.asarray(w, dtype=float) * CC.Units.RY_TO_CM)
    return np.array(rows)


def _build_qpath(dyn, job, n_points=320):
    """Build (qpath_raw, xaxis, xticks, xlabels) for phonon dispersion plotting."""
    uc = dyn.structure.unit_cell
    bp_str = job.get("bandpath")
    sp_pts = job.get("special_points")
    if bp_str and sp_pts:
        sp_clean = {str(k): [float(x) for x in v] for k, v in sp_pts.items()}
        qpath_raw, (xaxis, xticks, xlabels) = CC.Methods.get_bandpath(
            uc, str(bp_str), sp_clean, n_points=n_points
        )
        return qpath_raw, xaxis, xticks, xlabels

    atoms = dyn.structure.get_ase_atoms()
    bp = atoms.cell.bandpath(npoints=n_points)
    qpath = bp.cartesian_kpts()
    xaxis, xticks, xlabels = bp.get_linear_kpoint_axis()
    return qpath, xaxis, xticks, xlabels


def _detect_qpath_scale_factor(dyn, qpath):
    """Ensure qpath matches the exact unit convention of CC.ForceTensor.get_phonons_in_qpath."""
    if len(dyn.q_tot) < 2:
        return 1.0
    q_ref = np.array(dyn.q_tot[-1], dtype=float)
    w_exact, _ = dyn.DyagDinQ(len(dyn.q_tot) - 1)
    f_exact = np.sort(np.asarray(w_exact, dtype=float) * CC.Units.RY_TO_CM)

    best_mult = 1.0
    best_err = np.inf
    for mult in (1.0, 2.0 * np.pi, 1.0 / (2.0 * np.pi)):
        try:
            f_try = np.sort(CC.ForceTensor.get_phonons_in_qpath(dyn, [q_ref * mult])[0])
            err = float(np.max(np.abs(f_try - f_exact)))
            if err < best_err:
                best_err = err
                best_mult = mult
        except Exception:
            pass

    qtot_max = float(np.max(np.linalg.norm(np.array(dyn.q_tot), axis=1)))
    qpath_max = float(np.max(np.linalg.norm(np.array(qpath), axis=1)))
    if qtot_max > 1e-6 and qpath_max > 1e-6:
        ratio = qpath_max / qtot_max
        if abs(ratio / best_mult - 2.0 * np.pi) < 1.5:
            return 1.0 / (2.0 * np.pi)
        if abs(ratio / best_mult - 1.0 / (2.0 * np.pi)) < 0.08:
            return 2.0 * np.pi
    return 1.0


def _compute_dispersion_cm1(dyn, qpath):
    return CC.ForceTensor.get_phonons_in_qpath(dyn, qpath)


def write_final_summaries(
    job,
    dyn_harm,
    stage_dyns,
    dyn_final,
    dyn_hess,
    irred_idx,
    hess_freqs,
    hess_se,
    meta,
    csv_path="hessian_frequencies_summary.csv",
    txt_path="sscha_final_summary.txt",
):
    """Write machine-readable CSV and human-readable TXT final report across all stages."""
    f_harm = diag_irred_freqs_cm1(dyn_harm, irred_idx)
    f_stages = [(sname, diag_irred_freqs_cm1(sdyn, irred_idx)) for sname, sdyn in stage_dyns]
    f_fin = diag_irred_freqs_cm1(dyn_final, irred_idx)

    uc_fin = dyn_final.structure.unit_cell
    q_cart_list = [np.asarray(dyn_final.q_tot[iq], dtype=float) for iq in irred_idx]
    q_frac_list = [q @ uc_fin.T for q in q_cart_list]

    # 1. CSV
    header = ["q_irred_idx", "q_tot_idx", "qf_x", "qf_y", "qf_z", "mode", "harmonic_cm1"]
    for sname, _ in f_stages:
        header.append(f"{sname}_cm1")
    header.extend(["final_sscha_cm1", "hessian_cm1", "hessian_se_cm1"])

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i_ir, iq in enumerate(irred_idx):
            qf = q_frac_list[i_ir]
            n_modes = f_fin.shape[1]
            for m in range(n_modes):
                row = [
                    i_ir + 1, iq,
                    f"{qf[0]:.4f}", f"{qf[1]:.4f}", f"{qf[2]:.4f}",
                    m + 1,
                    f"{f_harm[i_ir, m]:.4f}",
                ]
                for _, fs in f_stages:
                    row.append(f"{fs[i_ir, m]:.4f}")
                row.append(f"{f_fin[i_ir, m]:.4f}")
                row.append(f"{hess_freqs[i_ir, m]:.4f}" if hess_freqs is not None else "")
                row.append(
                    f"{hess_se[i_ir, m]:.4f}"
                    if (hess_se is not None and not np.isnan(hess_se[i_ir, m]))
                    else ""
                )
                writer.writerow(row)

    # 2. Human-readable TXT summary
    lines = []
    lines.append("=" * 84)
    lines.append("SSCHA MULTI-STAGE FINAL ANHARMONIC SUMMARY | job=%s" % job.get("job_name", "sscha_job"))
    lines.append("=" * 84)
    lines.append("Temperature (K)      : %.2f" % float(job["temperature"]))
    lines.append("Target pressure (GPa): %s" % job.get("target_press_gpa", "fixed-volume"))
    lines.append("Supercell            : %s (%d atoms)" % (
        list(dyn_final.GetSupercell()),
        dyn_final.structure.N_atoms * int(np.prod(dyn_final.GetSupercell())),
    ))
    lines.append("")
    lines.append("--- Stage Execution Ledger ---")
    for st in meta.get("stages", []):
        kl_txt = (
            f", N_eff={st['neff']:.1f} (KL={st['kl']:.3f})"
            if st.get("neff") is not None else ""
        )
        lines.append(
            "  [%d] %-16s | N=%-3d max_pop=%-2d kpts=%-9s conv_thr=%-7g vc=%-5s | conv=%s%s" % (
                st["index"], st["name"], st["n_configs"], st["max_pop"],
                str(st["kpts"]), st["conv_thr"], str(st["vc_relax"]),
                str(st["converged"]), kl_txt,
            )
        )
    if meta.get("p_final_gpa") is not None:
        lines.append("")
        lines.append("Final pressure (GPa) : %.3f +/- %.3f (target=%s)" % (
            meta["p_final_gpa"], meta["p_final_err_gpa"],
            job.get("target_press_gpa", "fixed-V"),
        ))

    uc0 = dyn_harm.structure.unit_cell
    vol0 = abs(np.linalg.det(uc0))
    vol1 = abs(np.linalg.det(uc_fin))
    lines.append("")
    lines.append("--- Lattice Parameters (Angstrom) ---")
    lines.append("Initial cell lengths : a=%.6f, b=%.6f, c=%.6f | V=%.4f A^3" % (
        np.linalg.norm(uc0[0]), np.linalg.norm(uc0[1]), np.linalg.norm(uc0[2]), vol0))
    lines.append("Final   cell lengths : a=%.6f, b=%.6f, c=%.6f | V=%.4f A^3 (dV/V=%+.2f%%)" % (
        np.linalg.norm(uc_fin[0]), np.linalg.norm(uc_fin[1]), np.linalg.norm(uc_fin[2]),
        vol1, 100.0 * (vol1 - vol0) / vol0))

    lines.append("")
    lines.append("--- Lowest Non-Acoustic Modes per Irreducible q-Point (cm-1) ---")
    st_hdr = "".join(f" {('S'+str(i+1)):>8s}" for i in range(len(f_stages)))
    lines.append("%-5s %-22s | %8s%s | %-18s" % ("q_id", "q_frac", "Harm", st_hdr, "Hessian +/- SE"))
    lines.append("-" * 84)

    min_non_ac_hess = np.inf
    min_non_ac_se = np.nan
    min_non_ac_q = None
    for i_ir, iq in enumerate(irred_idx):
        qf = q_frac_list[i_ir]
        is_gamma = np.linalg.norm(q_cart_list[i_ir]) < 1e-6
        m0 = 3 if is_gamma else 0
        for m in range(m0, min(m0 + 3, f_fin.shape[1])):
            st_vals = "".join(f" {fs[i_ir, m]:8.2f}" for _, fs in f_stages)
            if hess_freqs is not None:
                hf = hess_freqs[i_ir, m]
                se = hess_se[i_ir, m] if hess_se is not None else np.nan
                h_str = f"{hf:8.2f} +/- {se:5.2f}" if not np.isnan(se) else f"{hf:8.2f} +/-   N/A"
                if hf < min_non_ac_hess:
                    min_non_ac_hess = hf
                    min_non_ac_se = se
                    min_non_ac_q = (i_ir + 1, qf, m + 1)
            else:
                h_str = "N/A"
            q_label = f"q{i_ir+1:d}m{m+1:d}"
            qf_str = f"({qf[0]:+.3f},{qf[1]:+.3f},{qf[2]:+.3f})"
            lines.append("%-5s %-22s | %8.2f%s | %-18s" % (
                q_label, qf_str, f_harm[i_ir, m], st_vals, h_str))

    lines.append("-" * 84)
    if min_non_ac_q is not None:
        if not np.isnan(min_non_ac_se):
            margin = min_non_ac_hess - 2.0 * min_non_ac_se
            if min_non_ac_hess < 0:
                verdict = "DYNAMICALLY UNSTABLE (imaginary Hessian mode < 0 cm-1)"
            elif margin > 0:
                verdict = "DYNAMICALLY STABLE (min_freq - 2*SE = %.2f > 0 cm-1)" % margin
            else:
                verdict = "MARGINAL / STATISTICALLY UNCERTAIN (min_freq=%.2f, SE=%.2f -> consider larger final n_configs)" % (
                    min_non_ac_hess, min_non_ac_se)
        else:
            verdict = "STABLE (min_freq=%.2f cm-1, SE not computed)" % min_non_ac_hess if min_non_ac_hess > 0 else "UNSTABLE"
        lines.append("Lowest non-acoustic Hessian mode: q%d %s mode %d = %.2f +/- %.2f cm-1" % (
            min_non_ac_q[0],
            f"({min_non_ac_q[1][0]:+.3f},{min_non_ac_q[1][1]:+.3f},{min_non_ac_q[1][2]:+.3f})",
            min_non_ac_q[2], min_non_ac_hess,
            min_non_ac_se if not np.isnan(min_non_ac_se) else 0.0,
        ))
        lines.append("Stability verdict               : %s" % verdict)
    lines.append("=" * 84)

    report = "\n".join(lines) + "\n"
    with open(txt_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(report)
    print("\n" + report, flush=True)


def plot_phonon_comparison(
    job,
    dyn_harm,
    dyn_final,
    dyn_hess,
    irred_idx,
    hess_freqs,
    hess_se,
    out_png="sscha_phonon_comparison.png",
):
    """300-DPI 2-panel figure: (a) softest mode per irred q + SE bars; (b) phonon dispersion."""
    f_harm = diag_irred_freqs_cm1(dyn_harm, irred_idx)
    f_fin = diag_irred_freqs_cm1(dyn_final, irred_idx)
    uc_fin = dyn_final.structure.unit_cell
    q_cart_list = [np.asarray(dyn_final.q_tot[iq], dtype=float) for iq in irred_idx]
    q_frac_list = [q @ uc_fin.T for q in q_cart_list]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.5, 5.2), dpi=300, gridspec_kw={"width_ratios": [1.0, 1.35]}
    )

    # Panel (a)
    x_pos = np.arange(len(irred_idx))
    q_labels, y_harm, y_sscha, y_hess, y_se = [], [], [], [], []
    for i_ir in range(len(irred_idx)):
        qf = q_frac_list[i_ir]
        is_gamma = np.linalg.norm(q_cart_list[i_ir]) < 1e-6
        m0 = 3 if is_gamma else 0
        q_labels.append(f"q{i_ir+1}\n({qf[0]:.1f},{qf[1]:.1f},{qf[2]:.1f})")
        y_harm.append(f_harm[i_ir, m0])
        y_sscha.append(f_fin[i_ir, m0])
        if hess_freqs is not None:
            y_hess.append(hess_freqs[i_ir, m0])
            y_se.append(0.0 if (hess_se is None or np.isnan(hess_se[i_ir, m0])) else hess_se[i_ir, m0])

    w_bar = 0.25
    ax1.bar(x_pos - w_bar, y_harm, width=w_bar, color="#7f8c8d", alpha=0.85, label="Harmonic (DFPT)")
    ax1.bar(x_pos, y_sscha, width=w_bar, color="#2980b9", alpha=0.88, label="SSCHA Auxiliary")
    if y_hess:
        ax1.bar(
            x_pos + w_bar, y_hess, width=w_bar, yerr=y_se, capsize=3.5,
            color="#c0392b", alpha=0.90, error_kw={"elinewidth": 1.2, "ecolor": "#2c3e50"},
            label="Free-Energy Hessian (+/- SE)",
        )
    ax1.axhline(0.0, color="black", lw=0.9, ls="--")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(q_labels, fontsize=8.5)
    ax1.set_ylabel(r"Lowest Non-Acoustic Frequency ($\mathrm{cm}^{-1}$)", fontsize=10.5)
    ax1.set_title("(a) Softest Mode per Irreducible q-Point", fontsize=11.5, fontweight="bold")
    ax1.legend(frameon=False, fontsize=8.5, loc="best")
    ax1.grid(axis="y", alpha=0.25)

    # Panel (b)
    try:
        qpath_raw, xaxis, xticks, xlabels = _build_qpath(dyn_final, job, n_points=320)
        scale = _detect_qpath_scale_factor(dyn_final, qpath_raw)
        qpath = np.asarray(qpath_raw, dtype=float) * scale

        disp_harm = _compute_dispersion_cm1(dyn_harm, qpath)
        disp_fin = _compute_dispersion_cm1(dyn_final, qpath)
        disp_hess = _compute_dispersion_cm1(dyn_hess, qpath) if dyn_hess is not None else None

        for b in range(disp_harm.shape[1]):
            lbl_h = "Harmonic (2x2x2 interp.)" if b == 0 else None
            lbl_s = "SSCHA Auxiliary" if b == 0 else None
            ax2.plot(xaxis, disp_harm[:, b], color="#7f8c8d", lw=1.0, ls="--", alpha=0.75, label=lbl_h)
            ax2.plot(xaxis, disp_fin[:, b], color="#2980b9", lw=1.1, alpha=0.75, label=lbl_s)
            if disp_hess is not None:
                lbl_he = "Free-Energy Hessian" if b == 0 else None
                ax2.plot(xaxis, disp_hess[:, b], color="#c0392b", lw=1.4, alpha=0.92, label=lbl_he)

        ax2.axhline(0.0, color="black", lw=0.9, ls=":")
        for xt in xticks:
            ax2.axvline(xt, color="#bdc3c7", lw=0.7, ls="-", zorder=0)
        pretty_labels = [r"$\Gamma$" if l in ("G", "Gamma") else l for l in xlabels]
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(pretty_labels, fontsize=10)
        ax2.set_xlim(xaxis[0], xaxis[-1])
        ax2.set_ylabel(r"Frequency ($\mathrm{cm}^{-1}$)", fontsize=10.5)
        ax2.set_title("(b) Anharmonic Phonon Dispersion (2x2x2 Supercell)", fontsize=11.5, fontweight="bold")
        ax2.legend(frameon=False, fontsize=8.5, loc="upper right")
        ax2.grid(axis="y", alpha=0.22)
    except Exception as exc:
        ax2.text(0.5, 0.5, f"Dispersion plot skipped:\n{exc}", ha="center", va="center", transform=ax2.transAxes)

    fig.suptitle(
        f"{job.get('job_name', 'SSCHA')} | T = {float(job['temperature']):.1f} K | P_target = {job.get('target_press_gpa', 'fixed-V')} GPa",
        fontsize=12.5, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"PLOT_SAVED: {out_png}", flush=True)


def plot_convergence_evolution(stage_evo_prefixes, out_png="sscha_convergence_evolution.png"):
    """Concatenate and plot Free Energy + Gradient evolution across all stages."""
    chunks = []
    labels = []
    for label, prefix in stage_evo_prefixes:
        path = prefix + ".dat"
        if not os.path.exists(path):
            continue
        try:
            arr = np.loadtxt(path)
            if arr.ndim == 1:
                arr = arr[None, :]
            if len(arr) > 0:
                chunks.append(arr)
                labels.append(label)
        except Exception:
            pass

    if not chunks:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 6.4), dpi=300, sharex=True)
    arr = np.vstack(chunks)
    steps = np.arange(1, len(arr) + 1)

    fe = arr[:, 0]
    fe_err = arr[:, 1] if arr.shape[1] > 1 else np.zeros_like(fe)
    ax1.errorbar(steps, fe, yerr=fe_err, fmt="-o", ms=3.5, lw=1.3, color="#2980b9", capsize=2.5)
    ax1.set_ylabel("SSCHA Free Energy (Ry)", fontsize=10)
    ax1.set_title("Multi-Stage SSCHA Convergence Evolution", fontsize=11.5, fontweight="bold")
    ax1.grid(alpha=0.25)

    if arr.shape[1] > 2:
        gfc = arr[:, 2]
        ax2.semilogy(steps, np.maximum(gfc, 1e-12), "-s", ms=3.5, lw=1.3, color="#c0392b", label="|grad FC|")
        if arr.shape[1] > 4 and np.any(arr[:, 4] > 0):
            ax2.semilogy(steps, np.maximum(arr[:, 4], 1e-12), "-^", ms=3.5, lw=1.2, color="#27ae60", label="|grad Struct|")
        ax2.set_ylabel("Gradient Norm (log)", fontsize=10)
        ax2.legend(frameon=False, fontsize=9)
    ax2.set_xlabel("Cumulative Minimization Step", fontsize=10)
    ax2.grid(alpha=0.25)

    # Mark stage transitions cleanly
    offset = 0
    y_top = ax1.get_ylim()[1]
    for i, (ch, lbl) in enumerate(zip(chunks, labels)):
        if i > 0:
            x_sep = offset + 0.5
            for ax in (ax1, ax2):
                ax.axvline(x_sep, color="#8e44ad", ls="--", lw=1.2, alpha=0.85)
        ax1.text(offset + 0.8, y_top, lbl, color="#8e44ad", fontsize=8.5, va="top", fontweight="bold")
        offset += len(ch)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"PLOT_SAVED: {out_png}", flush=True)


def run_all_postprocess(
    job,
    dyn_harm,
    stage_dyns,
    dyn_final,
    dyn_hess,
    irred_idx,
    hess_freqs,
    hess_se,
    meta,
    stage_evo_prefixes,
    out_dir=".",
):
    write_final_summaries(
        job=job,
        dyn_harm=dyn_harm,
        stage_dyns=stage_dyns,
        dyn_final=dyn_final,
        dyn_hess=dyn_hess,
        irred_idx=irred_idx,
        hess_freqs=hess_freqs,
        hess_se=hess_se,
        meta=meta,
        csv_path=os.path.join(out_dir, "hessian_frequencies_summary.csv"),
        txt_path=os.path.join(out_dir, "sscha_final_summary.txt"),
    )
    plot_phonon_comparison(
        job=job,
        dyn_harm=dyn_harm,
        dyn_final=dyn_final,
        dyn_hess=dyn_hess,
        irred_idx=irred_idx,
        hess_freqs=hess_freqs,
        hess_se=hess_se,
        out_png=os.path.join(out_dir, "sscha_phonon_comparison.png"),
    )
    plot_convergence_evolution(
        stage_evo_prefixes=stage_evo_prefixes,
        out_png=os.path.join(out_dir, "sscha_convergence_evolution.png"),
    )


def main():
    proj = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
    os.chdir(proj)
    with open("job.yaml", "r", encoding="utf-8") as f:
        job = yaml.safe_load(f)
    nqirr = int(job["nqirr"])
    out_prefix = job.get("output_dyn_prefix", "final_anharmonic_dyn_")

    dyn_harm = CC.Phonons.Phonons(job["init_dyn_prefix"], nqirr=nqirr)
    dyn_harm.Symmetrize()

    stage_dyns = []
    stage_evo = []
    stages_cfg = job.get("stages") or []
    for idx, st in enumerate(stages_cfg, start=1):
        sname = st.get("name", f"stage{idx}")
        spref = f"{sname}_dyn_"
        if os.path.exists(spref + "1"):
            stage_dyns.append((sname, CC.Phonons.Phonons(spref, nqirr=nqirr)))
        stage_evo.append((sname, f"sscha_evolution_{sname}"))
    if not stage_evo:
        stage_evo.append(("stage1", "sscha_evolution"))

    dyn_final = CC.Phonons.Phonons(out_prefix, nqirr=nqirr)
    dyn_hess = CC.Phonons.Phonons("hessian_" + out_prefix, nqirr=nqirr) if os.path.exists("hessian_" + out_prefix + "1") else None
    irred_idx = get_irred_q_indices(dyn_final)
    hess_freqs = diag_irred_freqs_cm1(dyn_hess, irred_idx) if dyn_hess is not None else None

    hess_se = None
    if os.path.exists("hessian_frequencies_summary.csv") and hess_freqs is not None:
        hess_se = np.full_like(hess_freqs, np.nan)
        with open("hessian_frequencies_summary.csv", "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("hessian_se_cm1"):
                    hess_se[int(row["q_irred_idx"]) - 1, int(row["mode"]) - 1] = float(row["hessian_se_cm1"])

    meta = {"stages": []}
    run_all_postprocess(
        job=job, dyn_harm=dyn_harm, stage_dyns=stage_dyns, dyn_final=dyn_final,
        dyn_hess=dyn_hess, irred_idx=irred_idx, hess_freqs=hess_freqs,
        hess_se=hess_se, meta=meta, stage_evo_prefixes=stage_evo, out_dir=".",
    )


if __name__ == "__main__":
    main()
