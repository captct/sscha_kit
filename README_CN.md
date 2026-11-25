# sscha_kit（自动化多阶段 SSCHA 非谐计算套件）

**支持 PBS/Torque 与 SLURM 的可迁移、动态工人池、四级断点续算 SSCHA 工作流**

[English](README.md) | [简体中文](README_CN.md)

> **版权与作者声明**：本软件版权归属 **吉林大学段德芳课题组**，由 **陈浩琦** 独立开发，采用 [MIT License](LICENSE) **完全开源免费使用**。

`sscha_kit` 基于 [`python-sscha`](https://github.com/SSCHAcode/python-sscha)、[`CellConstructor`](https://github.com/SSCHAcode/CellConstructor) 与 Quantum ESPRESSO（`pw.x`）构建，将传统多步手工操作封装为“**一次配置、全程自动跑完并出图验收**”的科研工具包。

---

## 核心特性

1. **动态工人池调度（`execution_mode: "worker_pool"`，“不用停、一直算”）**
   - 每代只提交 `max_nodes // nodes_per_job` 个工人作业，在共享锁目录通过原子 `mkdir` 动态抢占构型；
   - **零**构型间排队/轮询等待、**零**批次短板空等（快节点自动多算），并原生支持单作业**跨节点 MPI（`nodes_per_job >= 2`）**。
2. **多阶段由疏到密调度（`stages:`）**
   - 前序阶段用疏/中 $k$ 网格与较小样本快速硬化软模并收敛目标压强下的晶胞；
   - 末阶段自动切至密 $k$ 网格与大样本定胞精修，随后计算物理自由能 Hessian（$D^{(2)} + D^{(3)}$）及 **10 块刀切法（Jackknife）统计误差棒**（零额外 DFT 成本）。
   - 完整开放 `SSCHA_Minimizer` 参数（支持全局或分阶段覆盖）：`root_representation`、`preconditioning`、`min_step_dyn`（`lambda_a`）、`min_step_struc`（`lambda_w`）、`minim_struct`（`minim_struc`）、`minim_dyn`、`kong_liu_ratio` / `n_random_eff`、`meaningful_factor`、`gradi_op`、`max_ka`。
3. **四级全自动断点续算（`resume_from_stage: "auto"`）**
   - **阶段级**：已完阶段（`<stage>.DONE`）自动跳过并恢复准确代数编号；
   - **代数级**：当前阶段已完代数（`dyn_pop<K>_` + `ensembles_<stage>/`）自动继承；
   - **代内构型级**：若 100 个构型跑完 80 个时断电，重启后校验 `.pwo` 中的 `JOB DONE`，直接复用这 80 个结果，**只投工人补跑剩余 20 个**；
   - **终态 Hessian/出图级**：算 Hessian 或出图时中断，重启直接重载最后一代系综接续完成，零重复 DFT 计算。
4. **跨集群零修改迁移 & `/public` 全员共享**
   - `scripts/init_server.sh` 自动探测目标集群调度器（PBS/SLURM）、`pw.x` 路径、队列与节点核数并生成 `profiles.yaml`（含 `default` 档位）；`job.yaml` 只需写 `profile: "auto"`，迁移到新集群无需改动任务卡；
   - 启动脚本自动隔离各用户 `.bashrc` 中的 `CONDA_*` / `PYTHONPATH` 变量，避免多用户共享环境冲突；
   - **墙钟完全可选**：不写 `walltime` / `worker_walltime` 则不生成任何时间上限指令。
5. **全自动后处理与出图**
   - 自动输出 `hessian_frequencies_summary.csv`、`sscha_final_summary.txt`、`sscha_phonon_comparison.png`（300 DPI）与 `sscha_convergence_evolution.png`（300 DPI）。

---

## 一、安装与新集群部署

### 方式 A：在线安装依赖 + 共享部署
```bash
pip install -r requirements.txt
bash scripts/install_public.sh /public/apps/sscha_kit
```

### 方式 B：无外网集群离线部署
将打包好的 `sscha_env.tar.gz`（`conda-pack` 离线环境）放在仓库根目录后执行：
```bash
bash scripts/install_public.sh /public/apps/sscha_kit
```
脚本会自动完成：拷贝脚本、解压并 `conda-unpack` 重定位、运行 `scripts/init_server.sh` 探测本机队列并生成 `profiles.yaml`、设置全组 `a+rX` 权限。

*(可选)* 注册全局快捷命令，方便组员在任意目录直接输入 `sscha-check` / `sscha-run`：
```bash
ln -sf /public/apps/sscha_kit/check.sh /usr/local/bin/sscha-check
ln -sf /public/apps/sscha_kit/run.sh   /usr/local/bin/sscha-run
```
> 注意：运行账号需具备本机免密 `ssh localhost`（`sscha.Cluster` 通信自检所需）。

---

## 二、快速上手（3 步跑完一个体系）

在已准备好 `harmonic_dyn0..nqirr` 与 `./pseudo/*.UPF` 的体系目录下：

1. **复制并修改任务卡**：
   ```bash
   cp /public/apps/sscha_kit/job.yaml.example ./job.yaml
   vi job.yaml
   ```
   只需核对 `job_name`、`profile: "auto"`（或加 `queue: "<队列名>"`）、`max_nodes`、`nodes_per_job`、`temperature`、`target_press_gpa` 与 `pseudos`。
2. **非提交干跑自检**：
   ```bash
   sscha-check
   # 或：bash /public/apps/sscha_kit/check.sh
   ```
   看到 `ALL_CHECKS_PASSED (no job submitted)` 即可。
3. **一键启动**：
   - **登录节点后台守护**：`sscha-run`（或 `bash /public/apps/sscha_kit/run.sh`）
   - **批作业托管驱动**：
     ```bash
     # PBS
     qsub -q batch -l nodes=1:ppn=2 -v SSCHA_KIT_DIR=/public/apps/sscha_kit /public/apps/sscha_kit/run_in_job.sh
     # SLURM
     sbatch -p cpu -N 1 -n 2 --wrap="bash /public/apps/sscha_kit/run_in_job.sh job.yaml"
     ```

---

## 三、晶格弛豫与原子弛豫的平衡原理

在 `vc_relax: true` 阶段，每一代内部都自动执行双层耦合闭环：
1. **内层固定晶胞弛豫**：同时优化力常数（`min_step_dyn`）与原子平均坐标（`minim_struct: true`, `min_step_struc`）；
2. **外层晶胞步进**：根据非谐应力张量按 `bulk_modulus` 更新晶胞；
3. **双判据收敛**：只有原子+力常数梯度收敛与目标压强同时达标才结束本阶段。

标准三阶段建议为：**Stage 1 疏网变胞** $\to$ **Stage 2 中/密网变胞收敛晶格** $\to$ **Stage 3 同密网定胞精修（`vc_relax: false`）+ 自由能 Hessian**，确保最终 Hessian 位于严格一致的平衡晶格上。

---

## 四、自动产出文件

| 文件 | 说明 |
| :--- | :--- |
| `final_anharmonic_dyn_0..N` | 收敛后的辅助变分动力学矩阵（正定） |
| `hessian_final_anharmonic_dyn_0..N` | 物理自由能 Hessian 矩阵（含 $D^{(3)}$ 修正，用于判稳与算非谐 $T_c$） |
| `hessian_frequencies_summary.csv` | 全不可约 $q$ 点与各阶段频率演化表（含 Jackknife SE 误差棒） |
| `sscha_final_summary.txt` | 收敛账本、晶格/体积变化、终态压力与稳定性结论报告 |
| `sscha_phonon_comparison.png` | 300 DPI 双联图（最软模误差棒对比 + 傅里叶插值非谐声子色散） |
| `sscha_convergence_evolution.png` | 300 DPI 多阶段自由能与梯度收敛演化曲线 |

---

## 五、版权与开源协议

版权所有 © 2026 **吉林大学段德芳课题组**  
独立开发：**陈浩琦**  
本项目基于 [MIT License](LICENSE) **完全开源免费使用**。
