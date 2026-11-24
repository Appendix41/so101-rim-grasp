# 可以复现什么

| 路径 | 需要 | 是否重跑物理 |
|---|---|---|
| 从逐回合记录重算消融表 | Python 标准库 | 否，核对已有实放分数 |
| 从原始 RGB、已记录的 π0.5 像素提案重做放置选择；从训练终态重拟合放置模型 | CPU 和本仓库 | 否，选择完成后才查对应的实放后果 |
| 训练落点 LoRA | 单卡 CUDA、另行获取的 π0.5 基座与分词器；本仓库的配对训练图 | 否，更新预测像素的学生 |
| 重跑“侧抓 vs 夹沿”仿真 | 单独安装 Isaac Sim、Isaac Lab、leIsaac 及其 SO-101 资产 | 是，几何专家基线，无需 π0.5 |

仓库公开了结果审计所需的逐回合记录、配对观察包和训练代码；π0.5 基座、分词器、历史 LoRA 与原始仿真 HDF5 需单独获取。CPU 示例复算已发布的放置选择和消融计数；完整策略评测需载入对应权重并运行 Isaac Sim。新初始化训练会生成新的 LoRA；历史版本哈希记录在[评测协议](../evidence/dev-a-protocol.json)中。

## 1. 从仓库根目录运行：无需 GPU

```bash
python3 tools/audit.py
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/refit_landing.py
python tools/replay_choices.py
PYTHONPATH=src python -m unittest discover -s tests
```

`audit.py` 核对协议哈希、数据划分与抓取、松爪、稳定、分离、整底入布等判分字段。`refit_landing.py` 仅用训练集重建放置后果模型；`X`、`Y`、`scene` 数组与冻结模型逐元素相同。`replay_choices.py` 使用记录的 π0.5 提案重放 B 组 8 个持罐决策，并按已记录的执行结果统计选择表现；这三条 CPU 命令只读取公开数据，不会启动仿真或机械臂。

```text
natural: raw 7/8, wm 8/8, rule 8/8
probe: raw 8/32, wm 28/32, rule 32/32
```

## 2. 单卡后训练

LoRA 训练环境使用 Python 3.12、PyTorch 2.7.0+cu128、Transformers 5.5.4、PEFT 0.21.3，以及 [LeRobot 工作副本的确切提交](https://github.com/Seeed-Projects/lerobot/commit/4aaff99be4a1d81568c08c8f0296b41b40c99ec4)。Isaac Sim 使用独立仿真环境。训练环境按本机 CUDA 安装相容的 PyTorch、该提交的 `pi,peft` extras 和上述 CPU 依赖。

另行取得 [lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base/tree/22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0) 和 [PaliGemma tokenizer](https://huggingface.co/google/paligemma-3b-pt-224)；使用者自行遵守相应条款。`BASE` 与 `TOKENIZER` 指向本地目录，不需要在本仓库保存凭据。每条命令的 `--out` 必须是新目录。

```bash
export BASE=/path/to/pi05_base
export TOKENIZER=/path/to/paligemma-tokenizer
export HF_HUB_OFFLINE=1
python training/train_goal.py --stage distill --base "$BASE" --tokenizer "$TOKENIZER" --out outputs/v1
python training/train_goal.py --stage dagger --base "$BASE" --tokenizer "$TOKENIZER" --init outputs/v1 --out outputs/r1
```

第一条默认从新 rank-8 LoRA 开始训练落点，用 19 对原始示范、240 次更新。历史版则从黑布**动作** LoRA `47b6b912` 开始：持有这份检查点时可给第一条加 `--init /path/to/grasp-adapter`。第二条使用 19 旧 + 8 新摆位、192 步，只更新落点 LoRA；历史父权重为 `d7148f68`。`--steps 1` 可在新目录做一次前向、反向和保存检查。核心损失是归一化像素误差，加上同摆位换色两次预测的差分误差。归一化使用旧训练对，不读取开发包。

历史父权重与公开训练包的单步检查中，两次运行均抽到旧对 `13`、新对 `4`、第 `2` 帧（从 0 编号），首步损失均为 `0.6845837831497192`。[单步记录](../evidence/release-smoke.json)提供数据读取与训练更新的复核值。

放置 LoRA 输出图像像素目标；[cloth_goal.py](../src/cloth_goal.py)执行确定性读出、相机反投影与多帧确认，[visual_proposals.py](../src/visual_proposals.py)生成候选并调用几何搬放。抓取与放置后果模型分别见 [cloth_lift_wm.py](../src/cloth_lift_wm.py) 和 [cloth_landing_wm.py](../src/cloth_landing_wm.py)。

## 3. 在已配置的 Isaac 环境重跑几何抓取对照

原环境：Isaac Sim 5.1、Isaac Lab 2.3.0、leIsaac 0.4.0。以下只使用虚拟设备；请从能正常导入 `isaaclab` 和 `leisaac` 的仿真 Python 启动，机器人 USD 由 leIsaac 提供。

```bash
for approach in side rim; do
  python sim/grasp_sweep.py --headless --device cuda:0 --approach "$approach" \
    --heights-mm 60 90 120 --diameter-mm 100 --grasp-mm 73 \
    --repeats 3 --xy-jitter-mm 12 --seed 931 --out "outputs/paired73-$approach"
done
```

历史结果见[侧抓](../evidence/side-script.json)与[夹沿](../evidence/rim-script.json)。该命令运行几何专家抓取对照；接触求解器、驱动和资产版本会影响重跑数值。完整 π0.5 抓放和真机试验各有独立评测流程。
