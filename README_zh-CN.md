# TSML-ChainPINN

**面向瞬态空化重构的三阶段物理信息学习框架**

[English](README.md) · [模型与方程](docs/METHOD.md) · [数据准备](docs/DATA.md) · [训练说明](docs/TRAINING.md)

![模型架构](docs/assets/architecture.png)

本仓库依据 2026-10-18 论文修订稿整理，提供双分支模型、三阶段训练、ZGB 气相输运约束、原网格推理，以及可独立运行的演示和实验工具。原项目源码、历史权重均保留在原位置。

## 先跑通，再接入数据

在本仓库目录运行：

```bash
python -m pip install -e ".[viz]"
tsml doctor
tsml demo --output outputs/demo --plot
```

无需真实 CFD 文件即可走完整个流程，自动生成演示数据、训练日志、模型检查点、40 帧评估结果和图表。演示使用小模型与合成字段，明确标注为流程演示，不冒充 CFD 或论文精度。

![实际运行的演示](docs/assets/demo_summary.png)

## 正式模型

- 两分支均为 10 个残差块、512 宽度，64 组 Fourier 频带、8 个注意力头。
- 参数量 **15,766,021**；正式配置保持完整，不使用演示中的小网络。
- Adam，完整批次 **4096**，三阶段 **80/120/160** 轮。
- 学习率 **1e-3 / 5e-4 / 1e-4**，第三阶段前 20 轮仅训练相分支。
- 相阈值 0.90，混合损失权重 1.0/0.5/2.0，第三阶段相损失权重 14.0。
- 包含速度散度、守恒混合物动量和 ZGB 气相输运约束，按物理尺度恢复导数。
- 选参用 128/32 划分，最终用 160/40 划分；归一化仅来自拟合数据。

正式设置集中在 [configs/paper.json](configs/paper.json)。物性采用上一版的可修改默认值；论文没有完整规定的实现选择集中在 [PAPER_ALIGNMENT.md](docs/PAPER_ALIGNMENT.md)。

## 仓库功能

| 功能 | 入口 |
|---|---|
| 环境/GPU检查 | `tsml doctor` |
| 一键演示 | `tsml demo` |
| 原网格数据转换 | `tsml prepare` |
| 数据完整性检查 | `tsml inspect-data` |
| 三阶段训练与断点续训 | `tsml train` / `--resume` |
| 全网格评估 | `tsml evaluate` |
| 三随机种子实验 | `tsml experiment` |
| 均值与样本标准差汇总 | `tsml aggregate` |
| 训练/评估绘图 | `tsml plot` |
| Python 坐标推理 | `Surrogate.predict()` |

每个子命令均可加 `--help`。完整数据格式和命令见英文首页与 `docs/`。

## 开源与复现

提供 MIT 许可证、引用元数据、自动检查流程和测试。包中不塞入旧权重、原始大数据或论文图片；不把旧的 39/40 帧重新编号成论文的 200 帧。

已验证代码流程，尚未用正式 200 帧 CFD 数据完整重训。代码可用于新的训练和评估，不能据此声称已复现原论文数值。
