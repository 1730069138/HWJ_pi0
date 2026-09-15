import pathlib

import openpi.models.pi0_config as pi0_config
import openpi.training.weight_loaders as weight_loaders
import openpi.training.config as _config
import openpi.transforms as _transforms
import openpi.policies.aloha_policy as _aloha_policy
from openpi.training.config import TrainConfig, LeRobotAlohaDataConfig


def get_demo_configs() -> list[TrainConfig]:
    """Return a list of configs for training on demo dataset."""
    
    configs = [
        # Full fine-tuning config
        TrainConfig(
            name="pi0_demo_full",
            model=pi0_config.Pi0Config(),
            data=LeRobotAlohaDataConfig(
                repo_id="/home/jun/openpi/demo_data",
                assets=_config.AssetsConfig(
                    assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                    asset_id="trossen",  # Using trossen as default since your dataset seems to be arm-based
                ),
                # You can customize the default prompt based on your dataset's task
                default_prompt="Put mug cup on the plate",
                use_delta_joint_actions=False,  # 禁用delta动作计算以避免维度不匹配
                adapt_to_pi=False,  # 不使用adapt_to_pi，因为我们是单臂
                repack_transforms=_transforms.Group(
                    inputs=[
                        _transforms.RepackTransform(
                            {
                                "images": {"cam_high": "observation.image", "cam_left_wrist": "observation.wrist_image"}, 
                                "state": "observation.state",
                                "actions": "action",
                            }
                        )
                    ]
                ),
                action_sequence_keys=("action",),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi0_base/params"
            ),
            num_train_steps=30_000,
            batch_size=4,  # 设置为4以适配2个GPU设备
            fsdp_devices=2,  # 使用2个GPU设备
        ),
        
        # LoRA fine-tuning config
        TrainConfig(
            name="pi0_demo_lora",
            model=pi0_config.Pi0Config(
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora"
            ),
            data=LeRobotAlohaDataConfig(
                repo_id="/home/jun/openpi/demo_data",
                assets=_config.AssetsConfig(
                    assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                    asset_id="trossen",
                ),
                default_prompt="Put mug cup on the plate",
                use_delta_joint_actions=False,  # 禁用delta动作计算以避免维度不匹配
                adapt_to_pi=False,  # 不使用adapt_to_pi，因为我们是单臂
                repack_transforms=_transforms.Group(
                    inputs=[
                        _transforms.RepackTransform(
                            {
                                "images": {"cam_high": "observation.image", "cam_left_wrist": "observation.wrist_image"},
                                "state": "observation.state",
                                "actions": "action",
                            }
                        )
                    ]
                ),
                action_sequence_keys=("action",),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi0_base/params"
            ),
            num_train_steps=30_000,
            freeze_filter=pi0_config.Pi0Config(
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora"
            ).get_freeze_filter(),
            ema_decay=None,  # Turn off EMA for LoRA finetuning
            batch_size=2,  # 降低批次大小以适应GPU内存限制
            fsdp_devices=2,  # 使用2个GPU设备
        ),
        
        # Pi0.5 fine-tuning config
        TrainConfig(
            name="pi05_demo",
            model=pi0_config.Pi0Config(pi05=True),
            data=LeRobotAlohaDataConfig(
                repo_id="/home/jun/openpi/demo_data",
                assets=_config.AssetsConfig(
                    assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                    asset_id="trossen",
                ),
                default_prompt="Put mug cup on the plate",
                use_delta_joint_actions=False,  # 禁用delta动作计算以避免维度不匹配
                adapt_to_pi=False,  # 不使用adapt_to_pi，因为我们是单臂
                repack_transforms=_transforms.Group(
                    inputs=[
                        _transforms.RepackTransform(
                            {
                                "images": {"cam_high": "observation.image", "cam_left_wrist": "observation.wrist_image"},
                                "state": "observation.state",
                                "actions": "action",
                            }
                        )
                    ]
                ),
                action_sequence_keys=("action",),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_base/params"
            ),
            num_train_steps=30_000,
            batch_size=4,  # 设置为4以适配2个GPU设备
            fsdp_devices=2,  # 使用2个GPU设备
        ),
    ]
    
    return configs