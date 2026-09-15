# 自定义数据集LoRA微调指南

本文档详细介绍了如何使用自定义数据集对OpenPI模型进行LoRA微调，包括配置文件设置、数据映射和常见问题解决方法。

## 概述

OpenPI支持使用自定义数据集进行模型微调，特别是通过LoRA（Low-Rank Adaptation）技术，可以在较低的计算资源消耗下实现高效的模型适配。

## 准备工作

### 1. 数据集格式要求

自定义数据集需要符合LeRobot格式，包含以下结构：
```
dataset_folder/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       └── ...
└── meta/
    ├── info.json
    └── episodes.jsonl
```

### 2. 环境准备

确保已安装好OpenPI环境并下载了基础模型权重。

## 配置文件设置

### 1. 创建自定义配置

在`src/openpi/training/misc/demo_config.py`文件中创建针对自定义数据集的配置：

```python
def get_demo_configs() -> list[TrainConfig]:
    """Return a list of configs for training on demo dataset."""
    
    configs = [
        # LoRA fine-tuning config
        TrainConfig(
            name="pi0_demo_lora",
            model=pi0_config.Pi0Config(
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora"
            ),
            data=LeRobotAlohaDataConfig(
                repo_id="/home/jun/openpi/demo_data",  # 数据集路径
                assets=_config.AssetsConfig(
                    assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                    asset_id="trossen",
                ),
                default_prompt="Put mug cup on the plate",  # 默认指令
                use_delta_joint_actions=False,  # 禁用delta动作计算以避免维度不匹配
                adapt_to_pi=False,  # 不使用adapt_to_pi，适用于单臂机器人
                repack_transforms=_transforms.Group(
                    inputs=[
                        _transforms.RepackTransform(
                            {
                                "images": {
                                    "cam_high": "observation.image",           # 高处相机图像
                                    "cam_left_wrist": "observation.wrist_image"  # 腕部相机图像
                                }, 
                                "state": "observation.state",  # 状态信息
                                "actions": "action",          # 动作信息
                            }
                        )
                    ]
                ),
                action_sequence_keys=("action",),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi0_base/params"
            ),
            num_train_steps=10_000,
            freeze_filter=pi0_config.Pi0Config(
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora"
            ).get_freeze_filter(),
            ema_decay=None,  # Turn off EMA for LoRA finetuning
            batch_size=4,    # 批次大小
            fsdp_devices=2,  # 使用的GPU设备数
        ),
    ]
    
    return configs
```

### 2. 关键配置参数说明

- `repo_id`: 指向自定义数据集的本地路径
- `paligemma_variant`和[action_expert_variant](file:///home/jun/openpi/src/openpi/models/pi0_config.py#L145-L145): 使用LoRA变体以启用参数高效微调
- [use_delta_joint_actions](file:///home/jun/openpi/src/openpi/training/config.py#L227-L227): 对于单臂机器人数据集设为False以避免维度不匹配
- [repack_transforms](file:///home/jun/openpi/src/openpi/transforms.py#L474-L474): 重新映射数据字段以匹配模型期望的输入格式
- [freeze_filter](file:///home/jun/openpi/src/openpi/models/pi0_config.py#L173-L173): 冻结基础模型参数，仅训练LoRA适配器

## 数据字段映射

### 1. 字段映射原则

数据字段必须与模型期望的输入结构完全一致。常见的映射包括：
- 图像字段：从'observation.image'映射到'cam_high'
- 腕部图像：从'observation.wrist_image'映射到'cam_left_wrist'
- 状态信息：从'observation.state'映射到'state'
- 动作信息：从'action'映射到'actions'

### 2. 嵌套字段处理

对于嵌套字段，使用'/'作为分隔符进行展平处理。例如：
```python
"images": {
    "cam_high": "observation/image",  # 注意使用斜杠而非点号
    "cam_left_wrist": "observation/wrist_image"
}
```

## 运行训练

使用以下命令启动训练：

```bash
CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run scripts/train.py pi0_demo_lora --exp-name=my_demo_lora_experiment --resume --fsdp-devices=2 --batch-size=2
```

### 参数说明：
- `CUDA_VISIBLE_DEVICES=0,1`: 限制程序只能看到第0和第1号GPU设备
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`: 设置JAX可以使用的GPU内存比例
- `--fsdp-devices=2`: 明确指定使用2个设备进行训练

## 常见问题及解决方案

### 1. 字段映射错误

**错误信息**: KeyError: 'images.cam_high' 或类似字段缺失错误

**解决方案**:
- 检查数据集中的实际字段名称，查看`meta/info.json`和`meta/episodes.jsonl`
- 确保在[repack_transforms](file:///home/jun/openpi/src/openpi/transforms.py#L474-L474)中正确映射字段
- 注意字段名称区分大小写

### 2. 维度不匹配

**错误信息**: 广播错误或维度不匹配

**解决方案**:
- 检查数据集中的状态(state)和动作(action)维度
- 对于单臂机器人数据集，设置[use_delta_joint_actions](file:///home/jun/openpi/src/openpi/training/config.py#L227-L227)=False
- 确保[action_sequence_keys](file:///home/jun/openpi/src/openpi/training/config.py#L228-L228)与数据集中实际的动作字段名称一致

### 3. GPU设备配置问题

**错误信息**: Batch size must be divisible by the number of devices

**解决方案**:
- 确保batch_size能被fsdp_devices整除
- 使用`CUDA_VISIBLE_DEVICES`环境变量限制可见GPU设备
- 在命令行中明确指定`--fsdp-devices`参数

## 总结

通过合理配置数据映射和训练参数，可以成功使用自定义数据集对OpenPI模型进行LoRA微调。关键是要确保数据字段正确映射，设备配置合理，并根据数据集特点调整相关参数。