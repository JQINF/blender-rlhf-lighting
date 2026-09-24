# blender-rlhf-lighting

**Blender 影棚里的产品自动打光：NN 继承 → RL 两步精修 → RM 裁决 best-of-16 出图。**

English: [README.md](README.md)

在 Blender 摄影棚场景里，给定任意产品模型，自动产出电商白底图质量的布光方案：
嵌入检索把"最近邻产品的人工布光"继承为起点 → RLOO 策略做 2 步交互式精修
（第 1 步渲染预览图进观测，第 2 步出终值）→ 人标偏好模型（RM）对 16 个采样候选
裁决出 winner。附带完整训练管线（RM 训练 / RL 训练 / 盲标评估基建）。

## 方法一览

```
新品 ─▶ 嵌入最近邻继承（起点）
     ─▶ 策略 2 步精修（step1 渲预览图 → obs2；step2 渲终值图）
     ─▶ RM 裁决 best-of-16（部署） / RM z 分（训练奖励）
```

**策略训练器 = RLOO**（`scripts/train/rloo_continuous_action.py`）：

- **同起点成组采样**——`reset()` 抽一个起点（产品 / 机位 / 相机与产品抖动 / 继承配置+噪声），
  `reset_replay()` 把该起点逐字重放 K=10 次；组内唯一差异来自策略自身的 σ 采样
- **精确 leave-one-out 基线**——`advantage_i = R_i − mean(R_{j≠i})`。无 critic、无 GAE、无自举：
  2 步定长 episode + 终端 RM 奖励（r1 ≡ 0）下，价值函数只会带来一个可被钻的错误面
- **为什么不用 off-policy**——SAC 前任经五连修复把奖励从 −0.93 单调抬到 −0.49，却始终
  渐近于"什么都不做"的基线：replay 旧数据 + 自举 Q + actor 最大化 Q，会让 actor 专挑
  critic 的外推误差（实测 Q 估值 9~19 而真实回报 ≈ −2）。RLOO 保持 on-policy 并彻底去掉 critic
- **只要刹车，不要奖金**——clip 0.2 + target-KL 0.2 早停 + 梯度裁剪 0.5 + 塌缩预警链 +
  每 5 轮分段存 ckpt；σ 自学自退火，无熵奖励
- **吞吐**——两个常驻 Blender 渲染 worker 并行；46 起点 × K10 × 2 步 = 每轮 920 环境步
  （300k 步 ≈ 326 轮）

完整配方表、观测布局、阈值与评估协议见
[docs/method.md](docs/method.md) · [docs/reproduction.md](docs/reproduction.md)

## 效果

**新品（训练时无种子的人工精调布光，85 个产品池）**，确定性策略评估，
RM（DINOv2-L 骨干偏好模型，z 分）口径：

| 池 | Δ(策略 − 噪继承基线) | 胜格 | Δ(策略 − BC 监督基线) |
|---|---|---|---|
| 种子产品 32 机位 | **+1.674** | 31/32 | +0.446 |
| 留出产品 12 机位 | **+0.793** | 12/12 | +0.379 |
| 新品 85 机位 | **+1.200** | 79/85 | +0.780 |

**人眼盲评**（85 对逐对同机位，双盲）：终版策略 vs 历代冠军模型
**17/36 = 0.472 vs 0.528，统计平局（p≈0.90）**；其中 **49/85（58%）的对子标注者
判定两边不可区分**——新品上近六成机位与冠军模型的输出在人眼无差别。

### 新品池 — 噪继承基线 / 监督回归 / 本仓库终版 / 历代最强基线

<img src="images/new_products_grid.png" width="620" alt="新品对比网格：噪继承基线、监督回归、本仓库终版、历代最强基线">

### 种子产品 — 人工种子 / 噪继承基线 / 本仓库终版

<img src="images/seed_products_grid.png" width="600" alt="种子产品对比网格：人工种子、噪继承基线、本仓库终版">

## 10 分钟验证（无需训练）

1. 从 [Releases](https://github.com/JQINF/blender-rlhf-lighting/releases) 下载四个资产：
   `run38e_final.pt`（终版策略）、`rm_v19c.pt`（打分尺）、`stage_pack.zip`（影棚场景）、
   `samples.zip`（示例产品 + 嵌入）
2. 解包：`stage_pack.zip` → `scene/`，`samples.zip` → `model/`，两个 `.pt` → `logs/ckpt/`
3. 按 [docs/reproduction.md](docs/reproduction.md) 的环境节装依赖（uv/pip + Blender 5.x）
4. 命令行一键出图：

```bash
HF_HUB_OFFLINE=1 python scripts/deploy_light.py --sku <示例产品编号> --view front \
  --ckpt logs/ckpt/run38e_final.pt --rm-ckpt logs/ckpt/rm_v19c.pt
```

或在 Blender 里装 `addons/lighting_deploy` 插件，打开 `scene/stage.blend`，
点「自动打光」按钮（NN 继承 → 策略精修 → best-of-16 → winner 上场景）。

> 插件若报"定位不到项目根"：设环境变量 `LRL_PROJECT_ROOT=<本仓库路径>`。

## 仓库结构

```
scripts/        全量管线：env/ 渲染 worker 层、train/（rloo 训练器·RM 训练·BC·评估·盲标基建）、debug/ 诊断
addons/         lighting_rl（交互调光插件）+ lighting_deploy（部署演示插件）
seeds/          46 个手调布光种子（72 维 JSON：8 灯槽 × 9 参数，front/high 双机位各一套）
docs/           三篇双语：method / evaluation / reproduction（中文 xxx.md + 英文 xxx_EN.md）
script/         数据清洗工具
images/         效果对比图
```

完整数据规格与产品接入契约见 [docs/reproduction.md](docs/reproduction.md)。

## 方法与关键结论（TL;DR）

方法三段与 era 训练配方（RLOO × 时代尺 × 零药）详见 [docs/method.md](docs/method.md)。
三个有通用意义的发现：

1. **覆盖即泛化**：产品特异的布光偏好无法从任何产品观测预测（嵌入/空间贡献图、
   线性/非线性全灭；种子灯光 94.8% 方差为产品特异）——新品要打好光，唯一被证明
   的机制是"人工调一个、存一个种子"靠数量覆盖。
2. **裁决力 ≠ 奖励力**：偏好模型当裁判的改善不传导到当 RL 奖励的梯度侧。
3. **离策略三件套在此类任务结构上不可救**：SAC 五连修复（−0.93→−0.49）单调有效
   但渐近线瞄基线；去 critic 的 RLOO + 同起点组内基线根治。

## 资产与许可

- 代码：MIT（LICENSE）
- 权重（run38e_final.pt / rm_v19c.pt）与种子：随 MIT 一并授权，无任何担保
- 示例产品模型（samples.zip）：仅限演示用途；你自己的产品数据永远不需要离开本机
- **训练数据不随仓库分发**：RM 的对子标注与渲染图集均不含在内——仓库提供标注与训练工具，
  用 `label_server` 标自己的对子，即可训练属于你自己口味的 RM
- **自带模型验证**：按接入契约（`product` 集合 + 最长边 1.0m + bbox 居中 + 旋转已 apply）
  放入你自己的 `.blend`，用同一个 `rm_v19c.pt` + `run38e_final.pt` 即可跑通
  「继承 → 精修 → best-of-16 裁决」全链路

## 引用

```bibtex
@misc{blender-rlhf-lighting,
  title  = {blender-rlhf-lighting: RL-refined product lighting for Blender studios},
  year   = {2026},
  note   = {https://github.com/JQINF/blender-rlhf-lighting}
}
```
