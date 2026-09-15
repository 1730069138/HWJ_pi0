# 归一化统计

按照通用做法，我们的模型在策略训练和推理期间会对本体感知状态输入和动作目标进行归一化处理。用于归一化的统计数据是在训练数据上计算的，并与模型检查点一起存储。

## 重新加载归一化统计

当您在新数据集上微调我们的模型时，您需要决定是(A)重用现有的归一化统计还是(B)在您的新训练数据上计算新的统计。哪个选项对您更好取决于您的机器人和任务与预训练数据集中机器人和任务分布的相似性。下面，我们列出了每个模型可用的所有预训练归一化统计。

**如果您的目标机器人与这些预训练统计之一匹配，请考虑重新加载相同的归一化统计。** 通过重新加载归一化统计，您数据集中的动作将对模型来说更加"熟悉"，这可以带来更好的性能。您可以通过在训练配置中添加一个指向相应检查点目录和归一化统计ID的`AssetsConfig`来重新加载归一化统计，如下所示为`pi0_base`检查点的`Trossen`（即ALOHA）机器人统计：

```python
TrainConfig(
    ...
    data=LeRobotAlohaDataConfig(
        ...
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            asset_id="trossen",
        ),
    ),
)
```

有关重新加载归一化统计的完整训练配置示例，请参见[训练配置文件](https://github.com/physical-intelligence/openpi/blob/main/src/openpi/training/config.py)中的`pi0_aloha_pen_uncap`配置。

**注意：** 要成功重新加载归一化统计，重要的是您的机器人+数据集要遵循预训练中使用的动作空间定义。我们在下面详细描述了我们的动作空间定义。

**注意 #2：** 重新加载归一化统计是否有益取决于您的机器人和任务与预训练数据集中机器人和任务分布的相似性。我们建议始终尝试两种方法，重新加载和使用在新数据集上计算的新统计集进行训练（有关如何计算新统计的说明，请参见[主README](../README.md)），然后选择对您的任务效果更好的那个。

## 提供的预训练归一化统计

以下是所有我们提供的预训练归一化统计列表。我们为`pi0_base`和`pi0_fast_base`模型都提供这些统计。对于`pi0_base`，将`assets_dir`设置为`gs://openpi-assets/checkpoints/pi0_base/assets`，对于`pi0_fast_base`，将`assets_dir`设置为`gs://openpi-assets/checkpoints/pi0_fast_base/assets`。
| 机器人 | 描述 | 资产ID |
|-------|-------------|----------|
| ALOHA | 带平行夹爪的6自由度双臂机器人 | trossen |
| Mobile ALOHA | 安装在Slate底盘上的移动版ALOHA | trossen_mobile |
| Franka Emika (DROID) | 基于DROID设置的带平行夹爪的7自由度手臂 | droid |
| Franka Emika (非DROID) | 带Robotiq 2F-85夹爪的Franka FR3手臂 | franka |
| UR5e | 带Robotiq 2F-85夹爪的6自由度UR5e手臂 | ur5e |
| UR5e 双手 | 带Robotiq 2F-85夹爪的双手UR5e设置 | ur5e_dual |
| ARX | 带平行夹爪的双手ARX-5机器人臂设置 | arx |
| ARX 移动 | 安装在Slate底盘上的移动版双手ARX-5机器人臂设置 | arx_mobile |
| Fibocom 移动 | 带2个ARX-5手臂的Fibocom移动机器人 | fibocom_mobile |

## Pi0 模型动作空间定义

开箱即用，`pi0_base`和`pi0_fast_base`都使用以下动作空间定义（从机器人后方向工作区看时，左侧和右侧定义如下）：
```
    "dim_0:dim_5": "左臂关节角度",
    "dim_6": "左臂夹爪位置",
    "dim_7:dim_12": "右臂关节角度（仅限双手）",
    "dim_13": "右臂夹爪位置（仅限双手）",

    # 对于移动机器人：
    "dim_14:dim_15": "x-y底盘速度（仅限移动机器人）",
```

本体感知状态使用与动作空间相同的定义，除了移动机器人的底盘x-y位置（最后两个维度），我们不会将其包含在本体感知状态中。

对于7自由度机器人（例如Franka），我们使用动作空间的前7个维度进行关节动作，第8个维度用于夹爪动作。

Pi机器人的一般信息：
- 关节角度以弧度表示，位置零对应于每个机器人接口库报告的零位置，除了ALOHA，标准ALOHA代码使用稍有不同的约定（详见[ALOHA示例代码](../examples/aloha_real/README.md)）。
- 夹爪位置在[0.0, 1.0]范围内，0.0对应完全打开，1.0对应完全关闭。
- 控制频率为UR5e和Franka的20 Hz，以及ARX和Trossen（ALOHA）手臂的50 Hz。

对于DROID，我们使用原始的DROID动作配置，前7个维度使用关节速度动作，第8个维度使用夹爪动作+15 Hz的控制频率。