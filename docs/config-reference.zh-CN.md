# `config.yaml` 参数说明

本文档对应 `phonon-kit` 配置格式 `schema_version: 1`，说明当前版本实际支持的全部 YAML 参数。

## 基本规则

- 所有相对路径都以 `config.yaml` 所在目录为基准，不受执行 `ph` 命令时所在目录影响。
- 字段名严格检查：写错字段或加入未知字段会直接报错，不会静默忽略。
- YAML 布尔值请写成不带引号的 `true` 或 `false`；空值写成 `null`。
- 长度单位为 Å，能量单位为 eV，力单位为 eV/Å，压力单位为 GPa，频率单位为 THz，温度单位为 K。
- 修改 YAML、结构、启用的模型、VASP 模板或调度配置会改变任务指纹。已完成任务会新建编号；未完成任务不会与新配置混用。
- 修改后建议先执行 `ph validate config.yaml`，再执行 `ph run config.yaml`。

## 完整配置示例

```yaml
schema_version: 1

project:
  name: sio2
  runs_dir: runs

structure:
  file: inputs/POSCAR

relaxation:
  enabled: false
  method: dpa4
  variable_cell: true
  pressure_gpa: 0.0
  fmax_ev_angstrom: 0.01
  max_steps: 1000
  trajectory_interval: 10

phonon:
  supercell: [2, 2, 2]
  displacement_angstrom: 0.01
  primitive: auto
  symmetry_tolerance: 1.0e-5
  band:
    path: auto
    points_per_segment: 101
  mesh: [30, 30, 30]
  thermal:
    temperature_min_k: 0
    temperature_max_k: 1000
    temperature_step_k: 10
    imaginary_policy: exclude
    significant_imaginary_thz: -0.1

methods:
  dpa4:
    type: deepmd
    model: builtin:dpa4
    device: cuda:0
    head: null
    enabled: true

  dpa3:
    type: deepmd
    model: inputs/models/dpa3.pth
    device: cuda:0
    head: null
    enabled: false

  dft:
    type: vasp
    template_dir: inputs/vasp
    enabled: false
    executor:
      type: dpdispatcher
      machine: inputs/dispatcher/machine.json
      resources: inputs/dispatcher/resources.json
      command: "source /opt/intel/oneapi/setvars.sh && mpirun -n 16 vasp_std"
      clean_remote_after_success: false
```

## 顶层字段

| 字段 | 必填 | 可选值 | 说明 |
|---|---:|---|---|
| `schema_version` | 是 | 只能为 `1` | 配置格式版本。不是软件版本。 |
| `project` | 是 | mapping | 项目名称与运行目录。 |
| `structure` | 是 | mapping | 输入结构。 |
| `relaxation` | 否 | mapping | 生成位移前的可选 DeepMD 弛豫。省略时不弛豫。 |
| `phonon` | 否 | mapping | 有限位移、谱线、网格和热力学设置。省略时使用默认值。 |
| `methods` | 是 | mapping | 至少定义并启用一个力计算方法。 |

## `project`：项目与运行目录

| 字段 | 必填 | 默认值 | 说明 |
|---|---:|---|---|
| `name` | 是 | 无 | 任务名称，也是运行目录前缀，如 `sio2-001`。只能包含字母、数字、点、下划线和连字符，并且必须以字母或数字开头。 |
| `runs_dir` | 否 | `runs` | 所有运行目录的父目录。可写相对路径或绝对路径。 |

示例：

```yaml
project:
  name: quartz-p0
  runs_dir: runs
```

## `structure`：输入结构

| 字段 | 必填 | 默认值 | 说明 |
|---|---:|---|---|
| `file` | 是 | 无 | 输入结构路径。推荐使用 VASP POSCAR。程序通过 ASE 读取；若文件包含多帧，则读取最后一帧。 |

结构必须包含原子和三维非奇异晶胞。程序会将其视为三维周期性结构，并在运行目录中写出规范化 POSCAR。使用 VASP 时，`POTCAR` 元素顺序必须与结构中元素首次出现的顺序一致。

## `relaxation`：可选结构弛豫

弛豫使用指定 DeepMD 方法和 ASE `LBFGS`。`variable_cell: true` 时使用 `FrechetCellFilter` 同时优化原子位置和晶胞；否则只优化原子位置。弛豫未收敛时流程停止，不会继续生成声子位移。

| 字段 | 必填 | 默认值 | 可选值/限制 | 说明 |
|---|---:|---:|---|---|
| `enabled` | 否 | `false` | `true` / `false` | 是否在生成位移前弛豫。 |
| `method` | 否 | `dpa4` | 一个已启用的 DeepMD 方法名 | 指定弛豫所用模型，不能引用 VASP 方法。仅在 `enabled: true` 时生效。 |
| `variable_cell` | 否 | `true` | `true` / `false` | `true` 同时优化晶胞与原子；`false` 固定晶胞，只优化原子。 |
| `pressure_gpa` | 否 | `0.0` | 任意数值 | 变胞弛豫目标外压，正值表示压缩压力。固定晶胞时不参与优化。 |
| `fmax_ev_angstrom` | 否 | `0.01` | 大于 `0` | ASE 优化器的收敛阈值。值越小要求越严格、计算通常越久。 |
| `max_steps` | 否 | `1000` | 正整数 | 最大 LBFGS 优化步数。达到上限仍未收敛则停止流程。 |
| `trajectory_interval` | 否 | `10` | 正整数 | 每隔多少优化步保存一帧。轨迹始终包含初始帧和最终帧。 |

声子计算通常应使用充分弛豫的结构。如果输入已经可靠弛豫，可保持 `enabled: false`，避免不同方法比较时额外改变参考结构。

## `phonon`：有限位移和分析

### 位移超胞

| 字段 | 必填 | 默认值 | 可选值/限制 | 说明 |
|---|---:|---:|---|---|
| `supercell` | 否 | `[2, 2, 2]` | 三个正整数 | 沿三个晶格方向扩胞的倍数。当前只支持对角超胞，例如 `[3, 3, 2]`。超胞越大，单个位移计算越贵，但通常能包含更远程的力常数。 |
| `displacement_angstrom` | 否 | `0.01` | 大于 `0` | Phonopy 有限位移幅度。常见值约为 `0.01` Å；过小容易放大数值噪声，过大可能超出谐近似。 |
| `displacement_plusminus` | 否 | `auto` | `auto` / `true` / `false` | 是否为每个位移显式生成正、负两个构型。`auto` 沿用 Phonopy 默认判定。 |
| `displacement_diagonal` | 否 | `true` | `true` / `false` | 是否允许非笛卡尔轴向的对角位移。复现只使用 x/y/z 位移的旧数据时设为 `false`。 |
| `primitive` | 否 | `auto` | `auto` / `P` | `auto` 让 Phonopy 自动约化；`P` 使用恒等 primitive matrix，保留输入晶胞及其全部声子支。 |
| `symmetry_tolerance` | 否 | `1.0e-5` | 大于 `0` | Phonopy/spglib 对称性容差（Å）。它会影响识别到的对称性以及位移结构数量。不要仅为减少位移数量而随意调大。 |

`supercell` 是最重要的收敛参数之一。正式结果应检查增大超胞后关键频率和自由能是否基本不变。

### `phonon.band`：声子谱路径

| 字段 | 必填 | 默认值 | 可选值/限制 | 说明 |
|---|---:|---:|---|---|
| `path` | 否 | `auto` | `auto` 或 q 点列表 | `auto` 使用 SeeK-path；也可给出至少两个三维约化 q 点，相邻点依次连成路径。所有方法共享同一条路径。 |
| `points_per_segment` | 否 | `101` | 大于等于 `2` 的整数 | 每段高对称路径的采样点数。增大后曲线更平滑，但不会改变力计算数量。 |
| `labels` | 否 | 无 | 字符串列表 | 仅用于显式路径，数量必须和 `path` 中的 q 点相同。 |

### `mesh`：DOS 与热力学 q 网格

| 字段 | 必填 | 默认值 | 可选值/限制 | 说明 |
|---|---:|---:|---|---|
| `mesh` | 否 | `[30, 30, 30]` | 三个正整数 | 三个倒空间方向的 q 点网格，用于总声子 DOS 和热力学积分。网格越密，分析越慢、结果通常越平滑。 |

`mesh` 不会增加 DeepMD/VASP 位移超胞数量，只影响得到力常数之后的分析。正式自由能应进行网格收敛测试。

### `phonon.thermal`：振动热力学

| 字段 | 必填 | 默认值 | 可选值/限制 | 说明 |
|---|---:|---:|---|---|
| `temperature_min_k` | 否 | `0` | 大于等于 `0` | 输出温度下限。 |
| `temperature_max_k` | 否 | `1000` | 不小于最低温度 | 输出温度上限。 |
| `temperature_step_k` | 否 | `10` | 大于 `0` | 温度步长。 |
| `imaginary_policy` | 否 | `exclude` | 只能为 `exclude` | 热力学积分排除低于 `0 THz` 的虚频模式。 |
| `significant_imaginary_thz` | 否 | `-0.1` | 小于等于 `0` | “显著虚频”的报告阈值。最低频率低于该值时，结果标记 `thermodynamic_stability: false`。它不改变积分用的 `0 THz` 截断。 |

虚频被排除后仍可生成热力学表，但这只适合诊断。存在显著虚频时，不能把该自由能解释成严格稳定相的自由能。这里得到的是定体积谐振动自由能，包含零点能，但不包含静态能、电子自由能、`PV` 或 QHA 体积效应。

## `methods`：力计算方法

`methods` 的每个键是自定义方法名，例如 `dpa4`、`finetuned` 或 `dft`。方法名只能包含字母、数字、点、下划线和连字符，并以字母或数字开头。

可以同时启用多个 DeepMD 方法，但当前最多启用一个 VASP 方法。所有启用的方法共享同一套位移超胞和 q 点路径，便于直接比较。

### DeepMD 方法

```yaml
methods:
  dpa4:
    type: deepmd
    model: builtin:dpa4
    device: cuda:0
    head: null
    enabled: true
```

| 字段 | 必填 | 默认值 | 可选值/限制 | 说明 |
|---|---:|---:|---|---|
| `type` | 是 | 无 | `deepmd` | 方法类型。 |
| `model` | 是 | 无 | `builtin:dpa4` 或模型路径 | DeepMD ASE calculator 可加载的模型。支持 `.pt2`、`.pt`、`.pth`、`.pb`。路径可为相对或绝对路径。 |
| `device` | 否 | `cuda:0` | `cpu` 或 `cuda:N` | 推理设备，例如 `cuda:0` 使用第 0 张可见 GPU。模型本身必须支持所选设备；内置冻结 DPA4 只能使用 GPU。 |
| `head` | 否 | `null` | `null` 或模型支持的 head 名 | 多任务 checkpoint 的输出 head。`.pt2` 已固化 head，必须省略或写 `null`。 |
| `enabled` | 否 | `true` | `true` / `false` | 是否参与本次配置。禁用的方法不会验证、计算或比较。 |

文件扩展名只用于基础配置检查；模型是否真的兼容由 `ph validate` 的真实能量、力和应力推理决定。验证还会检查模型 type map 是否覆盖结构中的全部元素。

### VASP/DFT 方法

```yaml
methods:
  dft:
    type: vasp
    template_dir: inputs/vasp
    enabled: true
    executor:
      type: dpdispatcher
      machine: inputs/dispatcher/machine.json
      resources: inputs/dispatcher/resources.json
      command: "source /opt/intel/oneapi/setvars.sh && mpirun -n 16 vasp_std"
      clean_remote_after_success: false
```

| 字段 | 必填 | 默认值 | 可选值/限制 | 说明 |
|---|---:|---:|---|---|
| `type` | 是 | 无 | `vasp` | 方法类型。初版只支持 VASP 有限位移力计算。 |
| `template_dir` | 是 | 无 | 目录路径 | VASP 静态计算模板目录，必须包含 `INCAR`、`KPOINTS`、`POTCAR`。软件不会生成 POTCAR。 |
| `enabled` | 否 | `true` | `true` / `false` | 是否生成、提交、收集并分析该方法。 |
| `executor` | 是 | 无 | mapping | DPDispatcher 执行设置。 |

`INCAR` 必须是静态力计算：`IBRION = -1`、`NSW = 0`。建议使用严格电子收敛、高精度、`LREAL = .FALSE.`，并根据体系检查 ENCUT、KPOINTS、展宽和自旋设置。程序只检查明显错误，不会替你决定科学参数。

#### `methods.<name>.executor`

| 字段 | 必填 | 默认值 | 可选值/限制 | 说明 |
|---|---:|---:|---|---|
| `type` | 是 | 无 | 只能为 `dpdispatcher` | 当前唯一支持的调度后端。 |
| `machine` | 是 | 无 | JSON 文件路径 | DPDispatcher Machine 配置，例如 Bohrium 登录方式和远端环境。可能含凭据，不应提交 Git。 |
| `resources` | 是 | 无 | JSON 文件路径 | DPDispatcher Resources 配置，例如节点数、CPU/GPU 数、任务分组和轮询间隔。 |
| `command` | 是 | 无 | 非空字符串 | 每个位移任务在远端执行的命令。应负责加载环境并运行 `vasp_std`。 |
| `clean_remote_after_success` | 否 | `false` | `true` / `false` | 成功下载结果后是否允许 DPDispatcher 清理远端任务。为便于排错，建议先保持 `false`。 |

`ph init` 会生成 `machine.json.example` 和 `resources.json.example`。复制为不带 `.example` 的文件并填写实际参数。不同 DPDispatcher 后端的 JSON 字段由 DPDispatcher 决定，不属于本 YAML schema。

## 字段组合限制

- 至少定义一个方法，且至少一个方法的 `enabled` 为 `true`。
- `relaxation.enabled: true` 时，`relaxation.method` 必须引用一个已启用的 DeepMD 方法。
- `.pt2` 模型不能设置非空 `head`。
- 当前最多启用一个 VASP 方法，但可以启用任意多个 DeepMD 方法。
- `primitive` 可为 `auto` 或恒等矩阵标记 `P`；显式 `band.path` 使用约化 q 坐标。
- `imaginary_policy` 当前只能为 `exclude`。
- `supercell` 和 `mesh` 当前只接受三个正整数，不接受一般的 3×3 矩阵。
- `POTCAR` 中元素顺序必须与输入结构的元素顺序一致。

## 三个常用配置片段

只计算内置 DPA4：

```yaml
methods:
  dpa4:
    type: deepmd
    model: builtin:dpa4
    device: cuda:0
```

比较两个 DeepMD 模型：

```yaml
methods:
  dpa4:
    type: deepmd
    model: builtin:dpa4
    device: cuda:0
  finetuned:
    type: deepmd
    model: inputs/models/finetuned.pth
    device: cuda:0
    head: null
```

同一套位移比较 DPA 与 VASP：启用一个或多个 DeepMD 方法，再加入上文的 `dft` 方法。`ph run` 会先完成 DPA；DFT 异步提交后可用 `ph resume config.yaml` 查询。任务创建后会把结构、模型、VASP 模板和调度 JSON 复制到运行目录，因此案例输入修改后应使用 `ph resume runs/<run-id>` 精确恢复。`resume`、`collect` 和 `plot` 都接受运行目录或其中的 `config.resolved.yaml`。

## 配置检查

```bash
ph validate config.yaml
```

它会检查：

- YAML 字段、类型、范围和组合关系；
- 结构文件与三维晶胞；
- DeepMD 模型路径、type map、设备以及真实能量/力/应力推理；
- VASP 模板文件、静态计算关键设置与 POTCAR 元素顺序；
- DPDispatcher 的 Machine 和 Resources 配置。

验证通过只表示输入可运行，不代表超胞、q 网格、DFT 参数或结构已经达到所需的科学收敛精度。
