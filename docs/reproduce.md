# 可以复现什么

| 路径 | 需要 | 是否重跑物理 |
|---|---|---|
| 从逐回合记录重算消融表 | Python 标准库 | 否，核对已有实放分数 |
| 从原始 RGB、已记录的 π0.5 像素提案重做放置选择；从训练终态重拟合放置模型 | CPU 和本仓库 | 否，选择完成后才查对应的实放后果 |
| 训练落点 LoRA | 单卡 CUDA、另行获取的 π0.5 基座与分词器；本仓库的配对训练图 | 否，更新预测像素的学生 |
| 重跑“侧抓 vs 夹沿”仿真 | 单独安装 Isaac Sim、Isaac Lab、leIsaac 及其 SO-101 资产 | 是，几何专家基线，无需 π0.5 |

完整历史系统的抓取/落点 LoRA、上游基座、分词器和原始 HDF5 没有随本仓库分发。**不用它们就能复核公开放置决策，但不能把 CPU 重放说成重新完成了 16 次抓放。** 从新 LoRA 初始化运行训练示例也是一个新模型，不保证得到历史成绩。原始回退权重的哈希在[两个评测协议](../evidence/dev-a-protocol.json)里。

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

`audit.py` 检查协议哈希、训练/开发摆位分离，以及抓取、松爪、稳定、分离、整底入布等判分字段。`refit_landing.py` 只读取训练集，重建的 `X`、`Y`、`scene` 三个数组逐元素等于冻结模型。`replay_choices.py` 重放 B 组 8 个持罐决策；它不会调用 π0.5、不会借用专家标签给世界模型评分，也不会启动机械臂。

```text
natural: raw 7/8, wm 8/8, rule 8/8
probe: raw 8/32, wm 28/32, rule 32/32
```

## 2. 单卡后训练

项目训练进程使用 Python 3.12、PyTorch 2.7.0+cu128、Transformers 5.5.4、PEFT 0.21.3，以及 [LeRobot 工作副本的确切提交](https://github.com/Seeed-Projects/lerobot/commit/4aaff99be4a1d81568c08c8f0296b41b40c99ec4)。这不是仿真进程使用的 Python 环境。按本机 CUDA 安装相容的 PyTorch 后，在独立环境中安装该提交的 `pi,peft` extras，并安装上述 CPU 依赖。

另行取得 [lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base/tree/22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0) 和 [PaliGemma tokenizer](https://huggingface.co/google/paligemma-3b-pt-224)；使用者自行遵守相应条款。`BASE` 与 `TOKENIZER` 指向本地目录，不需要在本仓库保存凭据。每条命令的 `--out` 必须是新目录。

```bash
export BASE=/path/to/pi05_base
export TOKENIZER=/path/to/paligemma-tokenizer
export HF_HUB_OFFLINE=1
python training/train_goal.py --stage distill --base "$BASE" --tokenizer "$TOKENIZER" --out outputs/v1
python training/train_goal.py --stage dagger --base "$BASE" --tokenizer "$TOKENIZER" --init outputs/v1 --out outputs/r1
```

第一条默认从新 rank-8 LoRA 开始训练落点，用 19 对原始示范、240 次更新。历史版则从黑布**动作** LoRA `47b6b912` 开始：持有这份检查点时可给第一条加 `--init /path/to/grasp-adapter`。第二条使用 19 旧 + 8 新摆位、192 步，只更新落点 LoRA；历史父权重为 `d7148f68`。`--steps 1` 可在新目录做一次前向、反向和保存检查。核心损失是归一化像素误差，加上同摆位换色两次预测的差分误差。归一化使用旧训练对，不读取开发包。

发布时用历史父权重对公开训练包实跑了一次 DAgger 更新：两次都抽到旧对 `13`、新对 `4`、第 `2` 帧（从 0 编号），第一步损失都是 `0.6845837831497192`。[单步复核](../evidence/release-smoke.json)。这是数据打包和训练读出的检查，没有把它算作又一次完整后训练。

训练器输出像素适配器，**不能拿它输出的两个数驱动关节**。确定性读出、相机反投影和多帧确认在 [cloth_goal.py](../src/cloth_goal.py)；候选与几何搬放的实现见 [visual_proposals.py](../src/visual_proposals.py)；两个世界模型分别在 [cloth_lift_wm.py](../src/cloth_lift_wm.py)、[cloth_landing_wm.py](../src/cloth_landing_wm.py)。

## 3. 在已配置的 Isaac 环境重跑几何抓取对照

原环境：Isaac Sim 5.1、Isaac Lab 2.3.0、leIsaac 0.4.0。以下只使用虚拟设备；请从能正常导入 `isaaclab` 和 `leisaac` 的仿真 Python 启动，机器人 USD 由 leIsaac 提供。

```bash
for approach in side rim; do
  python sim/grasp_sweep.py --headless --device cuda:0 --approach "$approach" \
    --heights-mm 60 90 120 --diameter-mm 100 --grasp-mm 73 \
    --repeats 3 --xy-jitter-mm 12 --seed 931 --out "outputs/paired73-$approach"
done
```

历史结果：[侧抓](../evidence/side-script.json)与[夹沿](../evidence/rim-script.json)。接触求解器、驱动与资产版本可能影响重跑数值。此入口不是完整世界模型抓放评测，也不是实机部署脚本。
