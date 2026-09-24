# 复现手册

> English: [reproduction_EN.md](reproduction_EN.md)

## 环境

- **OS**：Linux（Ubuntu 系），GPU 可选（Cycles OPTIX→CUDA→CPU 自动回退，CPU 也能跑，慢）
- **Blender 5.x**：`blender` 在 PATH（snap/官方包均可）
- **Python 3.13 venv**：`uv sync`（按 pyproject.toml + uv.lock 建 venv 并装全部依赖：
  torch / torchvision / open_clip / transformers / gymnasium / numpy / Pillow / matplotlib）。
  开发环境用的是 CUDA 版 torch（cu126），CPU 也能跑（渲染走 Cycles，慢）
- **HF 模型**：RM 与嵌入骨干不随仓库分发（打分头之外的部分太大），首次使用需能访问
  HuggingFace（或 hf-mirror）拉取：
  - `rm_v19c.pt` 的骨干 = **`facebook/dinov2-large`（DINOv2-L/14，冻结，CLS token，权重 ~1.1GB）**，
    打分头（1024→128→1，约 13 万参数）在 ckpt 内
  - 嵌入骨干（ResNet18）走 torchvision 预训练权重
  - 拉过一次之后即可全程 `HF_HUB_OFFLINE=1` 离线运行
  - 例外：`train_rm.py --encoder clip_b16` 需要自备 `data/weights/open_clip_model.safetensors`
    （未随仓库分发）；其余骨干（含默认的 dinov2_l）都从 HF 拉取

## 资产就位（Releases 四件套）

```
stage_pack.zip ─▶ 解压出 stage.blend / embed_stage.blend → scene/
samples.zip    ─▶ 解压出 <sku>.blend → model/；<sku>.npy → data/embeddings/
run38e_final.pt / rm_v19c.pt ─▶ logs/ckpt/
```

## 验证终版模型（10 分钟）

```bash
# 部署口径一键出图（NN 继承 → 2 步精修 → best-of-16 → winner）
HF_HUB_OFFLINE=1 .venv/bin/python scripts/deploy_light.py --sku <示例sku> --view front \
  --ckpt logs/ckpt/run38e_final.pt --rm-ckpt logs/ckpt/rm_v19c.pt
```

或 Blender 插件路径：`addons/lighting_deploy` 安装 → 打开 `scene/stage.blend` →
N 面板「自动打光」。插件定位不到项目根时设 `LRL_PROJECT_ROOT=<仓库路径>`。

## 数据规格（接入自有产品）

- 产品工程：`<sku>.blend`，内含 `product` 集合；最长边 1.0m；bbox 中心在原点；
  旋转/缩放已 apply
- 嵌入：`HF_HUB_OFFLINE=1 .venv/bin/python scripts/extract_embeddings.py --all`
  （每产品 4 张 512² 辅助渲染，含 clay 三视图；改产品后必须重提）。
  渲染存在设备级非确定性（OPTIX/OIDN 像素抖动），跨设备/跨版本重提的嵌入会有 ~0.2% 量级差异——
  对继承检索与策略观测无实际影响（同机重跑的复现容差见 `scripts/debug/deploy_light_check.py`）
- 种子（可选但强烈建议）：`seeds/<sku>.json` = {front, high} 两套 72 维。
  **有种子的产品在部署时起点 = 自身种子（exclude-self 检索）**，即时受益于
  「覆盖即泛化」——这是唯一被证明的新品质量机制（见 docs/method.md）
- 渲染口径：训练/评估全链路 256²@25spp+OIDN；嵌入提取 512²；交付终图 2048²

## 训练复现（era 配方）

前提：产品池 ≥10 个（嵌入就位；种子可选但强烈建议）、同代 RM 就位（`scripts/train/train_rm.py`，
纯人标对子配方）。**注**：RM 的对子标注不随仓库分发——用 `label_server.py` 标自己的对子
（`make_blind200.py` 建批 → 浏览器标注 → `answers_to_labels.py` 入库），即可训练属于
你自己口味的 RM 并复现同一条配方。

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/train/rloo_continuous_action.py \
  --env blender --rm-ckpt logs/ckpt/rm_v19c.pt \
  --episodes-per-product 1 --num-envs 2 --num-starts 46 --rloo-k 10 \
  --total-timesteps 300000 --ent-coef 0.0 --min-grp-std 0.05 \
  --stop-after-warns 6 --ckpt-every 5 \
  --resume-actor logs/ckpt/run38e_final.pt \
  --out logs/ckpt/rloo_actor_repro.pt --log-file logs/rloo_repro.jsonl
```

- 每轮 920 步（46 起点 × K10 × 2 步），300k ≈ 326 updates；2 worker 并行 SPS 2-3，
  全程约 30h；`--resume-actor` 可断点续跑（换新 `--out` 与日志文件名）
- `--num-starts` 按你的产品池规模调（建议 = 池大小；每轮起点由无放回牌堆覆盖全池）
- 健康看板：`.venv/bin/python scripts/debug/read_train_log.py logs/rloo_repro.jsonl --follow`
  （z 窗应自起点 +0.3~0.5 缓爬至 +0.8 平台；grp_std 不腰斩；zero_var 不爬升）
- **两种起手**：① 在已发布策略上继续训（如上，热启动，改动最小、爬坡最快）；
  ② 从零训自己的风格：去掉 `--resume-actor` 改为随机初始化 + `--init-std 0.06`
  （σ 初值保险；这是冠军时代定下的 σ 管理口径）

## 评估与盲标

```bash
# 三池 z 对账（示例：新品池 85；BC 臂可选，给 --bc-ckpt 才跑）
HF_HUB_OFFLINE=1 .venv/bin/python scripts/train/eval_policy.py \
  --ckpt logs/ckpt/run38e_final.pt --rm-ckpt logs/ckpt/rm_v19c.pt \
  --skus 50,51,...,134 --episodes 85 --out-dir logs/rollout/repro_new

# 人尺盲评：label_server.py 起服务 → 浏览器标 data/pairs/<批>/ →
HF_HUB_OFFLINE=1 .venv/bin/python scripts/train/score_blind_eval.py \
  --blind-dir data/pairs/<批> --answers data/pairs/<批>/blind_eval_answers.json \
  --ckpt logs/ckpt/rm_v19c.pt
```

尺度纪律：任何数字必须带当时的 `--rm-ckpt`，跨尺不可比。
