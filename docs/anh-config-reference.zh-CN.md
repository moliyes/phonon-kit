# `anh.yaml` 三阶声子配置说明

本文对应 `phonon-kit 0.3.0`。三阶工作流只支持 DeepMD 力和 Phono3py 系统有限
位移；不负责结构弛豫、VASP、LBTE、同位素或边界散射。

## 完整示例

```yaml
schema_version: 1
project:
  name: sio2_anh
  runs_dir: runs
structure:
  file: inputs/POSCAR
  born_file: null
anharmonic:
  supercell: [2, 2, 2]
  fc2_supercell: null
  displacement_angstrom: 0.03
  primitive: auto
  symmetry_tolerance: 1.0e-5
  subtract_residual_forces: false
  mesh: [11, 11, 11]
  temperature_min_k: 100
  temperature_max_k: 1000
  temperature_step_k: 100
  lifetime_temperature_k: 300
  significant_imaginary_thz: -0.1
  continue_on_imaginary: true
methods:
  pretrained:
    type: deepmd
    model: builtin:dpa4
    device: cuda:0
  finetuned:
    type: deepmd
    model: inputs/models/finetuned.pt2
    device: cuda:0
```

所有相对路径以 `anh.yaml` 所在目录解析，未知字段直接报错。

## 字段

| 字段 | 默认值 | 含义与限制 |
|---|---|---|
| `schema_version` | 必填 | 只能为 `1`。 |
| `project.name` | 必填 | 运行前缀；输出为 `<name>-001`。 |
| `project.runs_dir` | `runs` | 运行目录。 |
| `structure.file` | 必填 | 已充分弛豫的三维周期结构。 |
| `structure.born_file` | `null` | 可选 Phonopy `BORN` 文件；提供后启用 NAC。Born 电荷必须与自动 primitive cell 一致。 |
| `anharmonic.supercell` | `[2,2,2]` | fc3 对角超胞，三个正整数。 |
| `anharmonic.fc2_supercell` | `null` | fc2 超胞；`null` 表示从 fc3 位移数据同时得到 fc2。二阶相互作用更长程时可设更大值。 |
| `anharmonic.displacement_angstrom` | `0.03` | 系统有限位移幅度，单位 Å。过小放大力噪声，过大引入高阶响应。 |
| `anharmonic.primitive` | `auto` | 当前只能为 `auto`。 |
| `anharmonic.symmetry_tolerance` | `1e-5` | 对称性识别容差，单位 Å；会改变位移数。 |
| `anharmonic.subtract_residual_forces` | `false` | 是否逐模型计算完美超胞力并从位移力中扣除。默认保留原始力。 |
| `anharmonic.mesh` | `[11,11,11]` | 三声子散射和热导率 q 网格。不会增加 DPA 位移数，但会显著增加散射计算成本。 |
| `temperature_min_k` | `100` | RTA 温度下限，必须大于 0 K。 |
| `temperature_max_k` | `1000` | 温度上限，必须等于 `min+n*step`。 |
| `temperature_step_k` | `100` | 正温度步长。 |
| `lifetime_temperature_k` | `300` | 输出 gamma/线宽/寿命明细的温度，必须位于上述网格。 |
| `significant_imaginary_thz` | `-0.1` | 显著虚频警告阈值，必须不大于 0。 |
| `continue_on_imaginary` | `true` | `true` 时带警告继续算 κ；`false` 时 fc2 检查后停止。 |

`methods` 至少启用一个 `type: deepmd` 方法。支持 `.pt2/.pt/.pth/.pb` 和
`builtin:dpa4`；设备为 `cpu` 或 `cuda:N`。`.pt2` 已固化 head，不能设置非空
`head`。模型是否真正兼容由 `ph anh run` 开始时的一次真实能量、力和应力推理
确认。

## 任务量与恢复

```bash
ph anh plan anh.yaml
ph anh run anh.yaml
ph anh status anh.yaml
```

系统三阶位移可能远多于二阶位移。`plan` 给出 fc3/fc2 位移数、超胞原子数、模型
数及总力评估数，但不加载模型。每个位移写入独立 NPZ；散射阶段每个不可约 q 点
写入 Phono3py 官方 HDF5。再次执行同一 `run` 只补缺失项。

若案例级结构、模型或 YAML 已改变，恢复旧任务应直接传运行目录：

```bash
ph anh run runs/sio2_anh-001
```

## 结果口径

- `gamma` 是 Phono3py 输出的三声子自能虚部，即半线宽，单位 THz。
- `linewidth = 2*gamma`。
- `lifetime_ps = 1/(4*pi*gamma_thz)`；`gamma<=0` 写成 `NaN`。
- `thermal_conductivity.csv` 给出六个张量分量和 `(xx+yy+zz)/3`，单位 W/m/K。
- 首版 κ 只含本征三声子 RTA 散射，不含同位素、边界、电子和四声子贡献。
- 出现显著虚频而继续计算时，结果只作诊断，不能据此宣称结构稳定。
- 多模型图表示 comparison/change/difference；没有 DFT 参考时不代表误差或精度。

正式结果至少检查 fc3 超胞、fc2 超胞、位移幅度和 q 网格收敛。极性体系若不提供
BORN/NAC，Γ 点附近的 LO–TO 分裂和群速度可能影响绝对热导率。
