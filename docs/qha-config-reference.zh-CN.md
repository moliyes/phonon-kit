# 多相 QHA 配置与使用说明

本文对应 `phonon-kit 0.2` 的 `ph qha` 工作流。普通单结构声子配置仍见
[config.yaml 参数说明](config-reference.zh-CN.md)。

## 1. 适用范围

QHA 工作流只比较相同最简化学组成的多晶型，例如 quartz、coesite 和 stishovite
都属于 SiO₂。它计算：

\[
G(P,T)=\min_V[U(V)+F_{\mathrm{vib}}(V,T)+PV].
\]

不同相可以使用不同大小的晶胞。Phonopy 内部使用 primitive-cell 体积和能量；
跨相比较前，程序自动除以 primitive cell 中的最简化学式单元数，得到
`eV/formula unit`。

本工作流不处理不同组成之间的化学势、凸包、液相、电子自由能或显式非谐效应。

## 2. 最快开始

```bash
ph qha init sio2_qha --phases quartz coesite stishovite
cd sio2_qha
```

将三个真实结构分别写入：

```text
inputs/phases/quartz/POSCAR
inputs/phases/coesite/POSCAR
inputs/phases/stishovite/POSCAR
```

初始化器提供的 POSCAR 只是可解析的 SiO₂ 占位结构，不能直接作为三个真实晶相
开展正式计算。

开始前先查看任务量：

```bash
ph qha plan qha.yaml
ph qha validate qha.yaml
ph qha run qha.yaml
```

`plan` 不创建 run、不运行模型，只估算每相的体积、有限位移数和 VASP 任务数。
实际固定体积弛豫可能轻微改变对称性，因此最终位移数以运行目录为准。

## 3. 完整 YAML

```yaml
schema_version: 1

project:
  name: sio2_qha
  runs_dir: runs

phases:
  quartz:
    structure: inputs/phases/quartz/POSCAR
    label: Quartz
    color: "#4c78a8"
    volume_ratios: [0.88, 0.91, 0.94, 0.97, 1.00, 1.03, 1.06]
    phonon:
      supercell: [2, 2, 2]

  coesite:
    structure: inputs/phases/coesite/POSCAR
    label: Coesite
    phonon:
      supercell: [2, 1, 2]
      mesh: [30, 30, 24]

volume_grid:
  ratios: [0.88, 0.91, 0.94, 0.97, 1.00, 1.03, 1.06]

volume_relaxation:
  fmax_ev_angstrom: 0.01
  max_deviatoric_stress_gpa: 0.1
  max_steps: 1000
  trajectory_interval: 10
  volume_tolerance_relative: 1.0e-5

phonon:
  supercell: [2, 2, 2]
  displacement_angstrom: 0.01
  symmetry_tolerance: 1.0e-5
  mesh: [30, 30, 30]
  significant_imaginary_thz: -0.1

qha:
  eos: vinet
  temperature:
    min_k: 0
    max_k: 1000
    step_k: 10
  pressure:
    min_gpa: 0
    max_gpa: 20
    step_gpa: 0.25
  imaginary_policy: exclude

methods:
  dpa4:
    type: deepmd
    model: builtin:dpa4
    device: cuda:0
    head: null
    enabled: true

  dft:
    type: vasp
    enabled: false
    templates:
      volume_relax: inputs/vasp/volume_relax
      static: inputs/vasp/static
      phonon: inputs/vasp/phonon
    phase_templates: {}
    executor:
      type: dpdispatcher
      machine: inputs/dispatcher/machine.json
      resources: inputs/dispatcher/resources.json
      command: "mpirun -n 16 vasp_std"
      clean_remote_after_success: false
```

所有相对路径均以 `qha.yaml` 所在目录解析。未知字段、带引号的伪布尔值和非法组合
会直接报错。

## 4. 参数说明

### `project`

| 字段 | 默认值 | 说明 |
|---|---:|---|
| `name` | 必填 | 运行目录前缀，只允许安全的文件名字符。 |
| `runs_dir` | `runs` | 版本化运行目录的父目录。 |

### `phases`

至少定义两个相。

| 字段 | 默认值 | 说明 |
|---|---:|---|
| `structure` | 必填 | 参考结构，推荐 POSCAR。所有相必须具有相同最简组成。 |
| `label` | 相的键名 | 图片和图例中的显示名。 |
| `color` | 自动 | Matplotlib 可识别的颜色；省略时自动分配。 |
| `volume_ratios` | 全局列表 | 针对该相覆盖全局体积比例。至少 5 个严格递增正数。 |
| `phonon` | 全局值 | 可逐相覆盖下面全部声子参数。 |

体积比例表示 `V/Vref`。晶格矢量的初始缩放因子为 `(V/Vref)^(1/3)`。输入结构
只负责提供体积中心和晶相拓扑，不要求程序先进行一次全变胞平衡弛豫；因此正式
计算必须确认 EOS 最低点被体积范围覆盖。

### `volume_grid`

`ratios` 是所有未单独覆盖晶相使用的体积比例列表。Phonopy QHA 至少需要 5 个
点；实际建议使用 7–11 个点，并根据目标压力在压缩侧提供足够范围。

### `volume_relaxation`

| 字段 | 默认值 | 单位/说明 |
|---|---:|---|
| `fmax_ev_angstrom` | `0.01` | 最大原子力阈值，eV/Å。 |
| `max_deviatoric_stress_gpa` | `0.1` | 最大偏应力分量阈值，GPa；不要求静水压力为零。 |
| `max_steps` | `1000` | DPA LBFGS 最大步数；VASP 使用模板中的 NSW。 |
| `trajectory_interval` | `10` | DPA 轨迹保存间隔；始终保留首尾帧。 |
| `volume_tolerance_relative` | `1e-5` | 目标体积与末态体积的最大相对误差。 |

DPA 通过 `FrechetCellFilter(constant_volume=True)` 同时优化内部坐标与晶胞形状。
VASP 通过 `ISIF=4` 完成相同物理约束。弛豫必须同时通过体积、力和偏应力检查，
否则不会继续使用该体积点。

### `phonon`

| 字段 | 默认值 | 说明 |
|---|---:|---|
| `supercell` | `[2,2,2]` | 有限位移超胞，只支持三个正整数。 |
| `displacement_angstrom` | `0.01` | 位移幅度，Å。 |
| `symmetry_tolerance` | `1e-5` | Phonopy/spglib 对称性容差。 |
| `mesh` | `[30,30,30]` | QHA 热力学积分 q 网格。 |
| `significant_imaginary_thz` | `-0.1` | 显著虚频报告阈值，THz。 |

这些字段均可在 `phases.<name>.phonon` 中逐相覆盖。不同相的晶胞大小不同，通常
应分别收敛超胞和 q 网格。

### `qha`

| 字段 | 默认值 | 说明 |
|---|---:|---|
| `eos` | `vinet` | 可选 `vinet`、`birch_murnaghan`、`murnaghan`。 |
| `temperature.min_k` | `0` | 温度下限，K。 |
| `temperature.max_k` | `1000` | 输出温度上限，K。 |
| `temperature.step_k` | `10` | 温度步长，必须整除范围。 |
| `pressure.min_gpa` | `0` | 压力下限，GPa。 |
| `pressure.max_gpa` | `20` | 压力上限，GPa。 |
| `pressure.step_gpa` | `0.25` | 压力步长，必须整除范围。 |
| `imaginary_policy` | `exclude` | 当前只能为 `exclude`。 |

程序会在温度上限外自动多计算一个点，供 Phonopy 数值微分使用；结果仍截止到
配置的 `max_k`。

负频模式以 `cutoff_frequency=0` 排除。如果任一体积存在低于显著阈值的模式，
计算继续，但该相和最终相图会标记 `thermodynamic_stability=false` 与
`imaginary_modes_excluded=true`。这不等于相已被证明稳定。

### DeepMD 方法

字段与普通声子配置一致：`model` 支持 `builtin:dpa4` 和 `.pt2/.pt/.pth/.pb`，
`device` 为 `cpu` 或 `cuda:N`，普通多任务 checkpoint 可设置 `head`，`.pt2` 不可
设置 head。每个方法独立弛豫自己的体积序列，避免混用不同势能面。

### VASP 方法与三套模板

每个模板目录必须含 `INCAR`、`KPOINTS`、`POTCAR`：

- `volume_relax`：要求 `ISIF=4`、`IBRION=1/2/3`、`NSW>0`。
- `static`：要求 `IBRION=-1`、`NSW=0`，用于读取最终 `e_0_energy`。
- `phonon`：要求 `IBRION=-1`、`NSW=0`，用于有限位移超胞的力。

三个阶段和所有相的 POTCAR 必须内容一致。程序不修改 INCAR，也不生成 POTCAR。
正式 EOS 应收敛 ENCUT、KPOINTS、展宽、电子收敛和 Pulay 应力。

某个相需要不同模板时，完整覆盖三个目录：

```yaml
phase_templates:
  coesite:
    volume_relax: inputs/vasp/coesite/volume_relax
    static: inputs/vasp/coesite/static
    phonon: inputs/vasp/coesite/phonon
```

`executor` 字段与普通 VASP 工作流一致。无 `--wait` 时先提交并快速返回；
`resume` 会重建相同 Submission 并防止重复任务。HTTP 5xx 上传/服务错误记录为
`retryable`，修复网络后直接再次 `resume`。

## 5. 运行、恢复和结果

```bash
ph qha run qha.yaml --only dpa4
ph qha run qha.yaml --only dpa4 dft --new
ph qha resume qha.yaml
ph qha resume qha.yaml --wait
ph qha collect qha.yaml
ph qha status qha.yaml
ph qha plot qha.yaml
```

配置及所有输入文件相同会恢复原 run；已完成后相同配置直接返回原结果。已完成后
修改配置会创建下一编号，未完成时修改配置则拒绝混用，除非显式 `--new`。

关键结果：

- `volume_points.csv`：每个体积的能量、体积、力、偏应力、最低频率和虚频比例。
- `e-v.dat`：Phonopy primitive-cell 单位的静态 E–V 数据。
- `qha_grid.csv/npz`：每个 P–T 点的 G、平衡体积、体积模量和有效性。
- `phase_map.csv`：稳定相、最低 Gibbs 能、次低相能差和警告标志。
- `phase_boundaries.csv`：相邻压力网格间通过 ΔG 线性插值得到的相界。
- `phase_diagram.png`、`gibbs_vs_pressure.png`、`gibbs_vs_temperature.png`。
- 多方法完成时生成 `results/comparison/method_agreement.csv/png`。

若某一相的拟合平衡体积超出采样区间，整个 P–T 点标为无数据，因为不能排除该相
在未覆盖体积处成为最低能相。相图中灰色区域表示这种情况；靠近采样边缘但尚未
外推的点在 `qha_grid.csv` 标记 `near_volume_edge=true`。

## 6. 科学检查清单

- 对每个相分别收敛有限位移超胞、q 网格、DPA/DFT 力精度和静态能量。
- 检查所有相的相对静态能偏置，而不只检查力 MAE。
- 确认目标压力范围对应的平衡体积在采样体积内。
- 检查明显虚频是否来自未充分弛豫、超胞不足、数值噪声或真实动力学不稳定。
- 极性材料若需要 LO–TO splitting，应独立评估 NAC 对自由能差和相界的影响；
  当前 QHA 工作流不自动计算 Born 电荷与介电张量。
- QHA 只包含由体积依赖频率描述的部分非谐效应；高温强非谐相需要更高阶方法。
