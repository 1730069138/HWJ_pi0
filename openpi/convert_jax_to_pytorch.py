#!/usr/bin/env python3
"""
将JAX训练的PI0检查点转换为PyTorch格式的脚本
"""

import subprocess
import sys
import os

def convert_jax_to_pytorch(jax_checkpoint_dir, output_dir, config_name="pi0_demo_lora", precision="bfloat16"):
    """
    将JAX检查点转换为PyTorch格式
    
    参数:
    jax_checkpoint_dir: JAX检查点所在目录
    output_dir: 输出PyTorch模型的目录
    config_name: 模型配置名称
    precision: 模型精度 (float32, bfloat16, float16)
    """
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建转换命令
    cmd = [
        "python3",
        "/home/jun/lerobot-mujoco-tutorial/openpi/examples/convert_jax_model_to_pytorch.py",
        "--checkpoint_dir", jax_checkpoint_dir,
        "--config_name", config_name,
        "--output_path", output_dir,
        "--precision", precision
    ]
    
    print(f"执行转换命令: {' '.join(cmd)}")
    
    # 执行转换
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("转换成功完成!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"转换过程中出现错误:")
        print(f"返回码: {e.returncode}")
        print(f"错误输出: {e.stderr}")
        return False

if __name__ == "__main__":
    # 默认路径设置
    JAX_CHECKPOINT_DIR = "/home/jun/openpi/checkpoints/pi0_demo_lora/my_demo_lora_experiment/29999"  # 你的JAX检查点路径
    OUTPUT_DIR = "/home/jun/lerobot-mujoco-tutorial/pytorch_models/pi0_converted"  # 输出路径
    CONFIG_NAME = "pi0_demo_lora"  # 默认配置名称
    PRECISION = "bfloat16"    # 默认精度
    
    # 如果提供了命令行参数，则使用它们
    if len(sys.argv) >= 3:
        JAX_CHECKPOINT_DIR = sys.argv[1]
        OUTPUT_DIR = sys.argv[2]
        
        # 可选的配置名称和精度参数
        CONFIG_NAME = sys.argv[3] if len(sys.argv) > 3 else "pi0_demo_lora"
        PRECISION = sys.argv[4] if len(sys.argv) > 4 else "bfloat16"
    else:
        print("未提供命令行参数，使用默认路径")
        print(f"JAX检查点路径: {JAX_CHECKPOINT_DIR}")
        print(f"PyTorch模型输出路径: {OUTPUT_DIR}")
        print("如需更改，请使用命令行参数:")
        print("用法: python convert_jax_to_pytorch.py <jax_checkpoint_dir> <output_dir> [config_name] [precision]")
        print("示例: python convert_jax_to_pytorch.py /home/user/jax_checkpoints /home/user/pytorch_model pi0_demo_lora bfloat16")
    
    success = convert_jax_to_pytorch(JAX_CHECKPOINT_DIR, OUTPUT_DIR, CONFIG_NAME, PRECISION)
    
    if success:
        print(f"\nPyTorch模型已保存到: {OUTPUT_DIR}")
    else:
        print("\n转换失败，请检查错误信息")
        sys.exit(1)