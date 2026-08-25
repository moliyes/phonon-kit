# phonon-kit

`phonon-kit` 是一个面向日常使用的有限位移声子、三阶非谐声子与准谐近似工作流。
它用 YAML 管理结构、DPA3/DPA4、VASP/DFT、声子谱、振动热力学、RTA 晶格热
导率、多相 QHA 和 P–T 相图，并把临时文件与最终结果严格分开。全局命令为 `ph`。

普通 `ph run` 处理一个结构的谐声子，`ph anh` 使用 DeepMD + Phono3py 处理三阶
声子，`ph qha` 处理相同组成的多个晶相。项目不实现 VASP-DFPT、变组成凸包、
四阶声子、熔化或模型训练。

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
Phono3py、ASE、SeeK-path、spglib、Matplotlib 和 DPDispatcher 会作为 Python 依赖安装。
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

三阶声子先预估任务量，再启动：

```bash
ph anh init sio2_anh
cd sio2_anh
ph anh plan anh.yaml
ph anh run anh.yaml
ph anh status anh.yaml
```

系统三阶位移数可能达到数千，务必先执行 `plan`。

运行产生：

```text
runs/sio2-001/
├── inputs/                 # 本次运行的结构、模型、VASP 和调度配置快照
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

当前配置格式全部字段、默认值、单位和组合限制见
[config.yaml 参数说明](docs/config-reference.zh-CN.md)。

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
ph resume runs/sio2-001     # 案例输入修改后，精确恢复指定运行
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
ph resume config.yaml       # 当前案例输入未改变时，查找匹配运行
ph resume runs/sio2-001     # 推荐：精确恢复指定运行
ph resume runs/sio2-001/config.resolved.yaml
ph resume runs/sio2-001 --wait
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
- 未完成时修改配置：拒绝混用；用具体运行目录恢复，或加 `--new` 新建版本。

从 0.2.1 开始，每个新运行会把实际使用的结构、模型、VASP 模板和
DPDispatcher 配置复制到运行内 `inputs/`，`config.resolved.yaml` 只引用这份
快照。之后修改案例级 `config.yaml` 或 `inputs/` 不会改变已有运行。恢复任务时
优先传具体运行目录；`collect` 和 `plot` 也接受运行目录。调度快照可能包含凭据，
请勿上传 `runs/`。

状态文件采用原子写入。失败详情保存在 `state.json` 和 `logs/`，软件不会制造
假的声子结果来掩盖失败。

独立执行 `ph validate` 的推理日志保存在案例目录的
`.phonon-kit/validation/logs/`，该目录默认被 Git 忽略。

## 7. 多相 QHA 与 P–T 相图

QHA 使用独立配置，不会改变前述单结构声子流程。快速创建三个 SiO₂ 晶相案例：

```bash
ph qha init sio2_qha --phases quartz coesite stishovite
cd sio2_qha
# 替换 inputs/phases/*/POSCAR
ph qha plan qha.yaml
ph qha validate qha.yaml
ph qha run qha.yaml
ph qha status qha.yaml
```

核心流程是每个方法、每个相分别执行多个固定体积点：

```text
固定体积优化原子和晶胞形状
→ 静态 U(V)
→ 每个体积的有限位移声子
→ Fvib(V,T)
→ G(P,T)
→ 多相最低 Gibbs 能比较
```

DPA 使用 `FrechetCellFilter(constant_volume=True)`；VASP 使用三套模板：
`volume_relax` 必须为 `ISIF=4`，`static` 和 `phonon` 必须为静态计算。程序不会
自动修改 INCAR，也不会生成 POTCAR。

```bash
ph qha run qha.yaml --only dpa4
ph qha run qha.yaml --only dft --new
ph qha resume qha.yaml
ph qha resume qha.yaml --wait
ph qha resume runs/sio2_qha-001
ph qha plot qha.yaml
```

不同晶相的 primitive cell 可能含不同数量的原子。程序先按 primitive cell 完成
Phonopy-QHA，再统一换算成 `eV/最简化学式单元` 比较。体积范围不足的 P–T 网格
点不会外推成稳定相，而是在图中显示灰色无数据区域。

若出现显著虚频，负频模式仍按 `cutoff_frequency=0` 排除并继续分析，但终端、
CSV、JSON 和图片都会标记 `thermodynamic_stability=false`。这类相图只能视为诊断
结果。

完整字段、VASP 模板、目录结构和结果含义见
[QHA 配置与使用说明](docs/qha-config-reference.zh-CN.md)。

## 8. 三阶非谐声子与 RTA 热导率

三阶工作流不弛豫结构，输入应已经可靠弛豫。所有模型共享相同结构、系统位移和
q 网格，因此适合比较预训练与微调模型带来的变化：

```text
POSCAR → 系统有限位移 → DeepMD 力 → fc2/fc3
       → 三声子散射 gamma → lifetime → RTA kappa(T)
```

默认 fc2 和 fc3 使用同一超胞；若二阶相互作用需要更长范围，可以设置独立的
`fc2_supercell`。`subtract_residual_forces` 默认关闭，与普通声子流程的原始力口径
一致。极性体系可通过 `structure.born_file` 提供 Phonopy BORN 文件启用 NAC。

```bash
ph anh run anh.yaml              # 相同未完成配置会自动续算
ph anh run runs/sio2_anh-001 # 案例输入改变后精确恢复
ph anh plot runs/sio2_anh-001
```

每个位移力和每个不可约 q 点的散射率都有 checkpoint。主要结果位于
`results/<method>/`：`fc2.hdf5`、`fc3.hdf5`、`kappa.hdf5`、热导率/寿命 CSV、
三张 PNG 和 `summary.json`。`gamma` 是半线宽，`linewidth=2*gamma`，寿命为
`1/(4*pi*gamma)` ps；非正 gamma 对应 `NaN` 寿命。

明显虚频默认不会阻断计算，但图表和摘要会标记为诊断结果。三阶首版只计算纯
三声子本征 RTA 热导率，不含同位素、边界、电子或四声子散射。超胞、位移幅度和
q 网格都必须做收敛检查。完整字段见
[三阶配置说明](docs/anh-config-reference.zh-CN.md)。

## 9. 常见问题

`ph validate` 提示 `.pt2` 无法加载：该冻结模型依赖 GPU、Torch/DeepMD 构建和
CUDA。先检查 `nvidia-smi`，并确保命令使用正确的 DeepMD Python。也可换用该
环境可加载的 `.pt/.pth/.pb`。

VASP 一直显示 waiting：检查每个 `disp-xxxx/vasprun.xml` 是否完整，而不仅是
OUTCAR 是否存在，然后执行 `ph resume` 或 `ph collect`。

自动高对称路径失败：通常是输入结构晶胞或对称性异常。先确保结构已弛豫、晶胞
非奇异，并适当调整 `symmetry_tolerance`。

自由能有警告：查看 `summary.json` 的最低频率、显著虚频比例和积分模式比例。

QHA 相图出现灰色区域：至少一个相在对应 P–T 点的拟合平衡体积超出采样体积
范围，或 EOS 无效。查看每相的 `volume_points.csv` 和 `qha_grid.csv`，扩大压缩侧
或膨胀侧体积范围后使用 `--new` 重新计算。

三阶热导率异常大或不平滑：先检查显著虚频、输入是否充分弛豫、`fc2/fc3`
超胞和 q 网格是否收敛；有限模型比较不能替代这些收敛性检查。

## 10. Hermes Agent Skill

项目在 `skills/materials-science/phonon-kit/` 内提供可版本化的 Hermes Skill，
用于配置、运行和诊断普通声子、三阶声子与多相 QHA 工作流。推荐从项目目录直接加载，避免
把 Skill 复制到用户目录后产生两份不同版本。

在 Hermes `config.yaml` 中保留已有配置并加入：

```yaml
skills:
  external_dirs:
    - /path/to/phonon-kit/skills
  config:
    phonon_kit:
      project_root: /path/to/phonon-kit
```

如果 `external_dirs` 或 `skills.config` 已存在，应在原列表和映射中追加，而不是
覆盖。新机器只需把两个路径换成实际项目目录，然后新建 Hermes 会话或执行
`/reload-skills`。可用下面的命令确认发现情况：

```bash
hermes skills list
hermes config get skills --json
```

显式调用示例：`/phonon-kit 检查这个 QHA 运行的体积覆盖和虚频警告`。自然语言
提及 Phonopy/Phono3py、DeepMD/VASP 声子、热导率、寿命、QHA 或 P–T 相图时也可自动触发。
Skill 不会因状态检查而擅自执行 `run`、`resume`、DFT 提交或模型推理。
