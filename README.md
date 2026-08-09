# phonon-kit

`phonon-kit` 是一个面向日常使用的有限位移声子工作流。它用一份 YAML 管理
结构、DPA3/DPA4、VASP/DFT、声子谱、DOS 和振动热力学，并把临时文件与最终
结果严格分开。全局命令为 `ph`。

初版只做谐近似、定体积的有限位移声子。它不做 QHA、相图、VASP-DFPT、DFT
弛豫或模型训练。

## 1. 安装

当前机器推荐直接复用 DeepMD 环境：

```bash
cd /home/begin/phonon-kit
./install.sh --with-test
ph --version
```

安装器优先选择当前可导入 DeepMD 的 Python，然后尝试
`/opt/deepmd-kit/.venv/bin/python`。也可明确指定：

```bash
./install.sh --python /path/to/deepmd/python
```

DeepMD 和 Torch 不会被安装器升级，避免破坏已经可用的 GPU 环境。Phonopy、
ASE、SeeK-path、spglib、Matplotlib 和 DPDispatcher 会作为 Python 依赖安装。
`requirements-verified.txt` 记录了本机完整验收时使用的依赖版本；它不负责安装
DeepMD/Torch。

内置 `models/dpa4_omat_neo704.pt2` 是 GPU 冻结模型。它不能在 CPU 上运行，也
不能设置 `head`。迁移到新机器后先执行 `ph validate`，以检查 CUDA、Torch、
DeepMD 和冻结模型是否匹配。

## 2. 最快用法

```bash
ph init sio2_case
cd sio2_case
ph validate config.yaml
ph run config.yaml
ph status config.yaml
```

`ph init` 会提供一个 SiO₂ 示例结构和使用内置 DPA4 的配置。默认不启用 VASP，
所以只要 GPU/DeepMD 可用即可运行。

运行产生：

```text
runs/sio2-001/
├── config.resolved.yaml
├── state.json
├── work/
│   ├── canonical/
│   ├── displacements/
│   ├── relaxation/
│   └── methods/
├── results/
│   ├── dpa4/
│   ├── dft/
│   └── comparison/
└── logs/
```

只需要长期保存 `config.yaml`、`inputs/` 和所需的 `runs/*/results/`；完整复现或
续跑时应保留整个 run 目录。

## 3. YAML 说明

路径均相对 `config.yaml` 解析。每份 YAML 对应一个结构，但可以同时定义多个
DPA 模型和一个 VASP 参考。

### 声子参数

```yaml
phonon:
  supercell: [2, 2, 2]          # 有限位移超胞
  displacement_angstrom: 0.01   # 位移幅度，Å
  primitive: auto
  symmetry_tolerance: 1.0e-5
  band:
    path: auto                   # SeeK-path 自动路径
    points_per_segment: 101
  mesh: [30, 30, 30]            # DOS 与热力学 q 网格
  thermal:
    temperature_min_k: 0
    temperature_max_k: 1000
    temperature_step_k: 10
    imaginary_policy: exclude
    significant_imaginary_thz: -0.1
```

所有方法复用同一个 `phonopy_disp.yaml` 和同一条 q 路径。更改超胞、位移、结构、
模型或模板后，配置指纹随之变化，不会把旧力数组混进新任务。

### DPA3/DPA4

```yaml
methods:
  dpa4:
    type: deepmd
    model: builtin:dpa4
    device: cuda:0

  dpa3:
    type: deepmd
    model: inputs/models/my_dpa3.pth
    device: cuda:0
    head: null
```

支持 DeepMD 可加载的 `.pt2`、`.pt`、`.pth`、`.pb`。`ph validate` 会实际计算
输入结构的一次能量、力和应力，并检查模型 type map 是否覆盖全部元素。模型能
否运行由真实推理决定，不依赖文件名判断 DPA3/DPA4。

每个位移保存在独立 `checkpoints/disp-xxxx.npz`。中断后再次运行：

```bash
ph resume config.yaml
```

只会补算缺失或损坏的位移。

### 可选 DPA 弛豫

```yaml
relaxation:
  enabled: true
  method: dpa4
  variable_cell: true
  pressure_gpa: 0.0
  fmax_ev_angstrom: 0.01
  max_steps: 1000
  trajectory_interval: 10
```

使用 `FrechetCellFilter + LBFGS`。弛豫模型必须是 `methods` 中启用的 DeepMD
方法。初始帧、每 10 步及最终帧写入 `relax.traj`；未收敛时不生成声子位移。
所有 DPA/DFT 方法共享同一个弛豫末态，便于严格比较。

## 4. VASP 与 DPDispatcher

把 `INCAR`、`KPOINTS`、`POTCAR` 放在一个模板目录。软件不会生成或分发
POTCAR。推荐的 INCAR 至少包含：

```text
PREC = Accurate
IBRION = -1
NSW = 0
EDIFF = 1E-8
LREAL = .FALSE.
```

在 YAML 增加：

```yaml
  dft:
    type: vasp
    template_dir: inputs/vasp
    executor:
      type: dpdispatcher
      machine: inputs/dispatcher/machine.json
      resources: inputs/dispatcher/resources.json
      command: "source /opt/intel/oneapi/setvars.sh && mpirun -n 16 vasp_std"
      clean_remote_after_success: false
```

可从 `ph init` 生成的 `.example` 文件复制并填写 DPDispatcher 配置。凭据文件、
POTCAR 和 runs 默认被 `.gitignore` 排除。

```bash
ph run config.yaml          # DPA 算完；VASP 提交后立即返回
ph status config.yaml
ph resume config.yaml       # 查询一次；完成时下载并分析
ph resume config.yaml --wait
```

每个 VASP 位移目录都保存 `atom-map.json`。程序用 Phonopy 官方 VASP parser 从
`vasprun.xml` 读取力，再显式恢复成规范 Phonopy 原子顺序。缺失或截断的 XML
不会被当成成功结果。若你用其他方式提交，只需把 `vasprun.xml` 放回对应
`work/methods/dft/jobs/disp-xxxx/`，然后运行 `ph collect config.yaml`。

## 5. 输出与物理含义

每个完成的方法自动生成：

- `forces.npy`、可用时的 `energies.npy`
- `FORCE_SETS`、`force_constants.hdf5`、`phonopy_params.yaml`
- `band.yaml`、`total_dos.dat`、`thermal_properties.yaml/csv`
- `phonon_band.png`、`phonon_band_dos.png`、`thermal_properties.png`
- `summary.json`

`thermal_properties.csv` 同时给出 Phonopy 摩尔单位和 eV/atom。自由能是谐振动
自由能，包含零点能，不包含结构静态能、电子自由能或 `PV` 项。

负频率按 Phonopy `cutoff_frequency=0` 排除。若存在低于
`significant_imaginary_thz` 的模式，图和 summary 会标记
`thermodynamic_stability: false`；此时自由能只适合作诊断，不能作为严格相稳定
结论。

当 DFT 和至少一个 DPA 都完成时，`results/comparison/` 自动生成：

- `metrics.csv`：力 MAE/RMSE、频率 MAE/RMSE、最低频率
- `phonon_band_compare.png`
- `force_comparison.png`
- `frequency_error.png`

## 6. 运行版本与恢复

- 第一次运行创建 `name-001`。
- 配置相同且任务未完成：`ph run` 自动恢复。
- 配置相同且任务已完成：返回原结果，不重复计算。
- 已完成后修改配置：自动创建 `name-002`。
- 未完成时修改配置：拒绝混用；使用原配置恢复，或加 `--new` 新建版本。

状态文件采用原子写入。失败详情保存在 `state.json` 和 `logs/`，软件不会制造
假的声子结果来掩盖失败。

独立执行 `ph validate` 的推理日志保存在案例目录的
`.phonon-kit/validation/logs/`，该目录默认被 Git 忽略。

## 7. 常见问题

`ph validate` 提示 `.pt2` 无法加载：该冻结模型依赖 GPU、Torch/DeepMD 构建和
CUDA。先检查 `nvidia-smi`，并确保命令使用正确的 DeepMD Python。也可换用该
环境可加载的 `.pt/.pth/.pb`。

VASP 一直显示 waiting：检查每个 `disp-xxxx/vasprun.xml` 是否完整，而不仅是
OUTCAR 是否存在，然后执行 `ph resume` 或 `ph collect`。

自动高对称路径失败：通常是输入结构晶胞或对称性异常。先确保结构已弛豫、晶胞
非奇异，并适当调整 `symmetry_tolerance`。

自由能有警告：查看 `summary.json` 的最低频率、显著虚频比例和积分模式比例。
QHA 与相图将在后续版本基于各 run 已保存的体积、组成、能量和温度网格实现。
