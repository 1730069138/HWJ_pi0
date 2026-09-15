# GPU设备配置与训练运行指南

本文档总结了在OpenPI项目中配置GPU设备和运行训练任务的相关问题及其解决方案。

## 问题描述

在尝试运行训练任务时，遇到了以下错误：
```
ValueError: Batch size 4 must be divisible by the number of devices 6.
```

这个错误表明虽然配置文件中设置使用2个GPU设备(`fsdp_devices=2`)，但系统实际检测到了6个GPU设备，并且当前的batch size(4)无法被6整除。

## 解决方案

### 1. 修改配置文件

首先，在配置文件`src/openpi/training/misc/demo_config.py`中确保batch_size能够被所使用的设备数整除：

```python
# LoRA fine-tuning config
TrainConfig(
    # ... 其他配置项 ...
    batch_size=4,  # 设置为4以适配2个GPU设备
    fsdp_devices=2,  # 使用2个GPU设备
),
```

### 2. 使用环境变量限制可见GPU设备

最关键的一点是使用`CUDA_VISIBLE_DEVICES`环境变量来限制JAX只能看到特定的GPU设备：

```bash
CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_demo_lora --exp-name=my_demo_lora_experiment --overwrite --fsdp-devices=2
```

这个命令做了以下几件事：
- `CUDA_VISIBLE_DEVICES=0,1`: 限制程序只能看到第0和第1号GPU设备
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`: 设置JAX可以使用的GPU内存比例
- `--fsdp-devices=2`: 明确指定使用2个设备进行训练

### 3. 综合配置原则

为了确保训练能够正确运行，需要遵循以下原则：

1. 在配置文件中正确设置`fsdp_devices`和`batch_size`，确保batch_size能被fsdp_devices整除
2. 在命令行中通过`--fsdp-devices`参数明确指定设备数量
3. 使用`CUDA_VISIBLE_DEVICES`环境变量限制JAX只能看到指定的GPU设备

## 总结

通过以上步骤，我们可以成功控制训练任务使用的GPU设备数量，解决了batch size与设备数量不匹配的问题。这种方法既保证了训练任务的正常运行，又提供了灵活的设备配置选项，让用户可以根据自己的硬件条件调整训练设置。