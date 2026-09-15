#!/usr/bin/env python3
"""
加载JAX模型并打印所有参数键，可选择转换为PyTorch格式。

此脚本使用orbax加载JAX模型检查点，可以：
1. 以分层结构打印所有参数键以供检查
2. 使用我们的PI0Pytorch模型将JAX模型转换为PyTorch格式

用法:
    # 仅检查键:
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /path/to/checkpoint --inspect_only
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /path/to/checkpoint --inspect_only

    # 转换为PyTorch:
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /path/to/checkpoint --output_path /path/to/output
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /path/to/checkpoint --output_path /path/to/output

示例:
    # pi0_droid
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi0_droid --output_path /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi0_droid_pytorch

    # pi0_aloha_sim
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim --output_path /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim_pytorch

    # pi05_droid
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi05_droid --output_path /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch
"""

import json
import os
import pathlib
import shutil
from typing import Literal

from flax.nnx import traversals
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch
import tyro

import openpi.models.gemma
import openpi.models.model
import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
from openpi.training import utils
import openpi.training.config as _config


def slice_paligemma_state_dict(state_dict, config):
    """将PaliGemma JAX参数转换为PyTorch格式。
    
    Args:
        state_dict: 包含JAX模型参数的字典
        config: 模型配置对象
        
    Returns:
        tuple: (处理后的主参数字典, 专家参数字典)
    """
    # 确定参数后缀（某些模型使用"value"后缀）
    suffix = "/value" if "img/embedding/kernel/value" in state_dict else ""

    # 处理图像嵌入层参数（卷积层）
    # JAX中卷积核的维度顺序与PyTorch不同，需要转置
    jax_key = f"img/embedding/kernel{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose(3, 2, 0, 1)

    # 处理图像嵌入层偏置
    jax_key = f"img/embedding/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.bias"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # 处理位置嵌入参数
    jax_key = f"img/pos_embedding{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.position_embedding.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).reshape(-1, config.vision_config.hidden_size)

    # 提取视觉编码器块的参数，准备进行切片处理
    # 层归一化参数
    encoderblock_layernorm0_scale = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_0/scale{suffix}")
    encoderblock_layernorm0_bias = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_0/bias{suffix}")
    encoderblock_layernorm1_scale = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_1/scale{suffix}")
    encoderblock_layernorm1_bias = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_1/bias{suffix}")

    # MLP层参数
    encoderblock_mlp_dense0_kernel = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel{suffix}")
    encoderblock_mlp_dense0_bias = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias{suffix}")
    encoderblock_mlp_dense1_kernel = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel{suffix}")
    encoderblock_mlp_dense1_bias = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias{suffix}")

    # 注意力机制参数
    encoderblock_attention_0_key_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel{suffix}"
    )
    encoderblock_attention_0_key_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias{suffix}"
    )
    encoderblock_attention_0_value_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel{suffix}"
    )
    encoderblock_attention_0_value_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias{suffix}"
    )
    encoderblock_attention_0_query_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel{suffix}"
    )
    encoderblock_attention_0_query_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias{suffix}"
    )
    encoderblock_attention_0_out_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel{suffix}"
    )
    encoderblock_attention_0_out_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias{suffix}"
    )

    # 遍历所有视觉编码器层，将参数转换为PyTorch格式
    for i in range(config.vision_config.num_hidden_layers):
        # 第一层归一化参数
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.weight"
        ] = encoderblock_layernorm0_scale[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.bias"
        ] = encoderblock_layernorm0_bias[i]
        
        # 第二层归一化参数
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.weight"
        ] = encoderblock_layernorm1_scale[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.bias"
        ] = encoderblock_layernorm1_bias[i]
        
        # MLP全连接层参数
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.weight"
        ] = encoderblock_mlp_dense0_kernel[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.bias"
        ] = encoderblock_mlp_dense0_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.weight"
        ] = encoderblock_mlp_dense1_kernel[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.bias"
        ] = encoderblock_mlp_dense1_bias[i]
        
        # 自注意力机制参数
        # Key投影矩阵
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.weight"
        ] = encoderblock_attention_0_key_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.bias"
        ] = encoderblock_attention_0_key_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        
        # Value投影矩阵
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.weight"
        ] = encoderblock_attention_0_value_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.bias"
        ] = encoderblock_attention_0_value_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        
        # Query投影矩阵
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.weight"
        ] = encoderblock_attention_0_query_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.bias"
        ] = encoderblock_attention_0_query_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        
        # 输出投影矩阵
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.weight"
        ] = encoderblock_attention_0_out_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.bias"
        ] = encoderblock_attention_0_out_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)

    # 编码器后层归一化参数
    jax_key = f"img/Transformer/encoder_norm/scale{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose()

    jax_key = f"img/Transformer/encoder_norm/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.bias"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # 多模态投影层参数
    jax_key = f"img/head/kernel{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose()

    jax_key = f"img/head/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # 文本解码器(Gemma)嵌入层参数
    jax_key = f"llm/embedder/input_embedding{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # 移除爱因斯坦求和注意力和MLP表示形式的参数
    llm_attention_attn_vec_einsum = state_dict.pop(f"llm/layers/attn/attn_vec_einsum/w{suffix}")
    llm_attention_kv_einsum = state_dict.pop(f"llm/layers/attn/kv_einsum/w{suffix}")
    llm_attention_q_einsum = state_dict.pop(f"llm/layers/attn/q_einsum/w{suffix}")

    llm_mlp_gating_einsum = state_dict.pop(f"llm/layers/mlp/gating_einsum{suffix}")
    llm_mlp_linear = state_dict.pop(f"llm/layers/mlp/linear{suffix}")

    llm_input_layernorm = state_dict.pop(f"llm/layers/pre_attention_norm/scale{suffix}")
    llm_post_attention_layernorm = state_dict.pop(f"llm/layers/pre_ffw_norm/scale{suffix}")

    # 处理语言模型层参数
    for i in range(config.text_config.num_hidden_layers):
        # Query投影权重重塑
        q_proj_weight_reshaped = (
            llm_attention_q_einsum[i]
            .transpose(0, 2, 1)
            .reshape(
                config.text_config.num_attention_heads * config.text_config.head_dim, config.text_config.hidden_size
            )
        )
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.q_proj.weight"] = (
            q_proj_weight_reshaped
        )

        # Key投影权重重塑
        k_proj_weight_reshaped = llm_attention_kv_einsum[i, 0, 0].transpose()
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.k_proj.weight"] = (
            k_proj_weight_reshaped
        )
        
        # Value投影权重重塑
        v_proj_weight_reshaped = llm_attention_kv_einsum[i, 1, 0].transpose()
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.v_proj.weight"] = (
            v_proj_weight_reshaped
        )

        # 输出投影权重重塑
        o_proj_weight_reshaped = (
            llm_attention_attn_vec_einsum[i]
            .transpose(2, 0, 1)
            .reshape(
                config.text_config.num_attention_heads * config.text_config.head_dim, config.text_config.hidden_size
            )
        )
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.o_proj.weight"] = (
            o_proj_weight_reshaped
        )

        # MLP门控投影权重
        gate_proj_weight = llm_mlp_gating_einsum[i, 0]
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.gate_proj.weight"] = (
            gate_proj_weight.transpose()
        )
        
        # MLP上投影权重
        up_proj_weight = llm_mlp_gating_einsum[i, 1]
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.up_proj.weight"] = (
            up_proj_weight.transpose()
        )
        
        # MLP下投影权重
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.down_proj.weight"] = (
            llm_mlp_linear[i].transpose()
        )
        
        # 输入层归一化权重
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.input_layernorm.weight"] = (
            llm_input_layernorm[i]
        )
        
        # 注意力后层归一化权重
        state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.post_attention_layernorm.weight"
        ] = llm_post_attention_layernorm[i]

    # 最终归一化层参数
    jax_key = f"llm/final_norm/scale{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.language_model.norm.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # 专家相关参数提取（包括pi05密集层参数）
    expert_dict = {}
    final_state_dict = {}

    # 需要提取的专家键列表（包括pi05的Dense层参数）
    expert_keys = [
        f"llm/final_norm_1/scale{suffix}",
        f"llm/final_norm_1/Dense_0/bias{suffix}",
        f"llm/final_norm_1/Dense_0/kernel{suffix}",
        f"llm/layers/attn/attn_vec_einsum_1/w{suffix}",
        f"llm/layers/attn/kv_einsum_1/w{suffix}",
        f"llm/layers/attn/q_einsum_1/w{suffix}",
        f"llm/layers/mlp_1/gating_einsum{suffix}",
        f"llm/layers/mlp_1/linear{suffix}",
        f"llm/layers/pre_attention_norm_1/scale{suffix}",
        f"llm/layers/pre_attention_norm_1/Dense_0/bias{suffix}",
        f"llm/layers/pre_attention_norm_1/Dense_0/kernel{suffix}",
        f"llm/layers/pre_ffw_norm_1/scale{suffix}",
        f"llm/layers/pre_ffw_norm_1/Dense_0/bias{suffix}",
        f"llm/layers/pre_ffw_norm_1/Dense_0/kernel{suffix}",
    ]

    # 分离专家参数和普通参数
    for key, value in state_dict.items():
        if key not in expert_keys:
            # 普通参数转换为PyTorch张量
            final_state_dict[key] = torch.from_numpy(value)
        else:
            # 专家参数保持numpy数组格式
            expert_dict[key] = value

    return final_state_dict, expert_dict


def slice_gemma_state_dict(state_dict, config, *, num_expert, checkpoint_dir, pi05):
    """将Gemma JAX参数转换为PyTorch格式。
    
    Args:
        state_dict: 包含JAX模型参数的字典
        config: Gemma模型配置
        num_expert: 专家编号
        checkpoint_dir: 检查点目录路径
        pi05: 是否为pi05模型的布尔值
        
    Returns:
        dict: 转换后的PyTorch参数字典
    """
    # 如果配置中缺少某些属性，则添加默认值
    if not hasattr(config, "vocab_size"):
        config.vocab_size = 257152  # PALIGEMMA_VOCAB_SIZE
    if not hasattr(config, "hidden_size"):
        config.hidden_size = config.width
    if not hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = config.depth
    if not hasattr(config, "num_attention_heads"):
        config.num_attention_heads = config.num_heads

    # 确定参数后缀
    suffix = "/value" if f"llm/layers/attn/attn_vec_einsum_{num_expert}/w/value" in state_dict else ""

    # 提取注意力和MLP层参数
    llm_attention_attn_vec_einsum = state_dict.pop(f"llm/layers/attn/attn_vec_einsum_{num_expert}/w{suffix}")
    llm_attention_kv_einsum = state_dict.pop(f"llm/layers/attn/kv_einsum_{num_expert}/w{suffix}")
    llm_attention_q_einsum = state_dict.pop(f"llm/layers/attn/q_einsum_{num_expert}/w{suffix}")

    llm_mlp_gating_einsum = state_dict.pop(f"llm/layers/mlp_{num_expert}/gating_einsum{suffix}")
    llm_mlp_linear = state_dict.pop(f"llm/layers/mlp_{num_expert}/linear{suffix}")

    # 检查是否有Dense层（用于pi05/自适应归一化）或scale层（用于常规pi0）
    if "pi05" in checkpoint_dir:
        # Pi05使用自适应归一化
        llm_input_layernorm_bias = state_dict.pop(f"llm/layers/pre_attention_norm_{num_expert}/Dense_0/bias{suffix}")
        llm_post_attention_layernorm_bias = state_dict.pop(f"llm/layers/pre_ffw_norm_{num_expert}/Dense_0/bias{suffix}")
        llm_input_layernorm_kernel = state_dict.pop(
            f"llm/layers/pre_attention_norm_{num_expert}/Dense_0/kernel{suffix}"
        )
        llm_post_attention_layernorm_kernel = state_dict.pop(
            f"llm/layers/pre_ffw_norm_{num_expert}/Dense_0/kernel{suffix}"
        )
    else:
        # 常规pi0使用标准RMSNorm
        llm_input_layernorm = state_dict.pop(f"llm/layers/pre_attention_norm_{num_expert}/scale{suffix}")
        llm_post_attention_layernorm = state_dict.pop(f"llm/layers/pre_ffw_norm_{num_expert}/scale{suffix}")

    # 处理每一层的参数
    for i in range(config.num_hidden_layers):
        # Query投影权重重塑
        q_proj_weight_reshaped = (
            llm_attention_q_einsum[i]
            .transpose(0, 2, 1)
            .reshape(config.num_attention_heads * config.head_dim, config.hidden_size)
        )
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.q_proj.weight"] = (
            q_proj_weight_reshaped
        )

        # Key投影权重重塑
        k_proj_weight_reshaped = llm_attention_kv_einsum[i, 0, 0].transpose()
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.k_proj.weight"] = (
            k_proj_weight_reshaped
        )
        
        # Value投影权重重塑
        v_proj_weight_reshaped = llm_attention_kv_einsum[i, 1, 0].transpose()
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.v_proj.weight"] = (
            v_proj_weight_reshaped
        )

        # 输出投影权重重塑
        o_proj_weight_reshaped = (
            llm_attention_attn_vec_einsum[i]
            .reshape(config.num_attention_heads * config.head_dim, config.hidden_size)
            .transpose(1, 0)
        )
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.o_proj.weight"] = (
            o_proj_weight_reshaped
        )

        # MLP门控投影权重
        gate_proj_weight = llm_mlp_gating_einsum[i, 0]
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.gate_proj.weight"] = (
            gate_proj_weight.transpose()
        )
        
        # MLP上投影权重
        up_proj_weight = llm_mlp_gating_einsum[i, 1]
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.up_proj.weight"] = (
            up_proj_weight.transpose()
        )
        
        # MLP下投影权重
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.down_proj.weight"] = llm_mlp_linear[
            i
        ].transpose()

        # 根据模型类型处理层归一化参数
        if "pi05" in checkpoint_dir:
            # Pi05使用自适应归一化 - 直接使用Dense层参数
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.dense.bias"] = (
                llm_input_layernorm_bias[i]
            )
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.dense.bias"] = (
                llm_post_attention_layernorm_bias[i]
            )
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.dense.weight"] = (
                llm_input_layernorm_kernel[i].transpose()
            )
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.dense.weight"] = (
                llm_post_attention_layernorm_kernel[i].transpose()
            )
        else:
            # 常规pi0使用标准RMSNorm
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.weight"] = (
                llm_input_layernorm[i]
            )
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.weight"] = (
                llm_post_attention_layernorm[i]
            )

    # 处理最终归一化层
    if "pi05" in checkpoint_dir:
        # Pi05使用自适应归一化 - 直接使用Dense层参数
        final_norm_bias = state_dict.pop(f"llm/final_norm_{num_expert}/Dense_0/bias{suffix}")
        final_norm_kernel = state_dict.pop(f"llm/final_norm_{num_expert}/Dense_0/kernel{suffix}")
        state_dict["paligemma_with_expert.gemma_expert.model.norm.dense.bias"] = final_norm_bias
        state_dict["paligemma_with_expert.gemma_expert.model.norm.dense.weight"] = final_norm_kernel.transpose()
    else:
        # 常规pi0使用标准RMSNorm
        state_dict["paligemma_with_expert.gemma_expert.model.norm.weight"] = state_dict.pop(
            f"llm/final_norm_{num_expert}/scale{suffix}"
        )

        # state_dict["paligemma_with_expert.gemma_expert.lm_head.weight"] = embedding_vector # 权重是绑定的.

    # 转换所有参数为PyTorch张量
    final_state_dict = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            final_state_dict[key] = torch.from_numpy(value)
        else:
            final_state_dict[key] = value

    return final_state_dict


def slice_initial_orbax_checkpoint(checkpoint_dir: str, restore_precision: str | None = None):
    """通过首先使用JAX模型加载器恢复来加载和处理参数。
    这尊重在模型恢复期间发生的dtype转换。
    
    Args:
        checkpoint_dir: 检查点目录路径
        restore_precision: 恢复精度设置
        
    Returns:
        dict: 包含处理后参数的字典
    """
    # 使用仓库恢复工具加载纯参数字典（移除value后缀）
    params = openpi.models.model.restore_params(
        f"{checkpoint_dir}/params/", restore_type=np.ndarray, dtype=restore_precision
    )

    # 返回扁平化的参数映射
    return {"paligemma_params": traversals.flatten_mapping(params["PaliGemma"], sep="/"), "projection_params": params}


def load_jax_model_and_print_keys(checkpoint_dir: str):
    """
    从检查点加载JAX模型并打印所有参数键。
    
    Args:
        checkpoint_dir: 检查点目录路径
    """
    # 处理检查点目录路径
    checkpoint_dir = os.path.abspath(checkpoint_dir) if not checkpoint_dir.startswith("gs://") else checkpoint_dir
    
    # 初始化检查点器
    checkpointer = ocp.PyTreeCheckpointer()
    metadata = checkpointer.metadata(f"{checkpoint_dir}/params")
    
    # 打印模型参数信息
    print(utils.array_tree_to_info(metadata))


def convert_pi0_checkpoint(
    checkpoint_dir: str, precision: str, output_path: str, model_config: openpi.models.pi0_config.Pi0Config
):
    """
    将PI0 JAX检查点转换为PyTorch格式。
    
    Args:
        checkpoint_dir: JAX检查点路径
        precision: 模型精度 (float32, bfloat16, float16)
        output_path: 保存转换后PyTorch模型的路径
        model_config: 模型配置
    """
    print(f"正在将PI0检查点从 {checkpoint_dir} 转换到 {output_path}")
    print(f"模型配置: {model_config}")

    # 通过JAX恢复分解orbax检查点，以尊重dtype
    initial_params = slice_initial_orbax_checkpoint(checkpoint_dir=checkpoint_dir, restore_precision="float32")

    # 处理投影参数
    if model_config.pi05:
        # Pi05模型的投影层键
        keys = [
            "action_in_proj",
            "action_out_proj",
            "time_mlp_in",
            "time_mlp_out",
        ]
    else:
        # 常规模型的投影层键
        keys = [
            "state_proj",
            "action_in_proj",
            "action_out_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        ]

    # 提取投影参数
    projection_params = {}
    for key in keys:
        kernel_params = initial_params["projection_params"][key]["kernel"]
        bias_params = initial_params["projection_params"][key]["bias"]
        if isinstance(kernel_params, dict):
            weight = kernel_params["value"]
            bias = bias_params["value"]
        else:
            weight = kernel_params
            bias = bias_params

        pytorch_weight_key = f"{key}.weight"
        pytorch_bias_key = f"{key}.bias"

        # 转换参数为PyTorch格式（转置权重矩阵）
        projection_params[pytorch_weight_key] = torch.from_numpy(np.array(weight)).T
        projection_params[pytorch_bias_key] = torch.from_numpy(np.array(bias))

    # 创建基于检查点路径的配置
    # 所有模型使用相同的PaliGemma配置结构
    class PaliGemmaConfig:
        """PaliGemma模型配置类"""
        def __init__(self):
            # 视觉配置
            self.vision_config = type(
                "obj",
                (object,),
                {
                    "hidden_size": 1152,           # 隐藏层大小
                    "num_hidden_layers": 27,       # 隐藏层数量
                    "num_attention_heads": 16,     # 注意力头数
                    "intermediate_size": 4304,     # 中间层大小
                    "patch_size": 14,              # 图像块大小
                    "projection_dim": 2048,        # 投影维度
                },
            )()
            
            # 文本配置
            self.text_config = type(
                "obj",
                (object,),
                {
                    "hidden_size": 2048,           # 隐藏层大小
                    "num_hidden_layers": 18,       # 隐藏层数量
                    "num_attention_heads": 8,      # 注意力头数
                    "head_dim": 256,               # 头维度
                    "intermediate_size": 16384,    # 中间层大小
                },
            )()

    # 创建PaliGemma配置实例
    paligemma_config = PaliGemmaConfig()
    
    # 获取动作专家配置
    action_expert_config = openpi.models.gemma.get_config("gemma_300m")

    # 处理PaliGemma权重
    paligemma_params, expert_params = slice_paligemma_state_dict(initial_params["paligemma_params"], paligemma_config)

    # 从expert_params处理Gemma权重
    gemma_params = slice_gemma_state_dict(
        expert_params, action_expert_config, num_expert=1, checkpoint_dir=checkpoint_dir, pi05=model_config.pi05
    )

    # 实例化PyTorch模型
    pi0_model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_config)

    # 合并所有参数（我们的模型结构不需要前缀）
    all_params = {**paligemma_params, **gemma_params, **projection_params}

    # 加载状态字典
    pi0_model.load_state_dict(all_params, strict=False)

    # 根据指定精度设置模型精度
    if precision == "float32":
        pi0_model = pi0_model.to(torch.float32)
    elif precision == "bfloat16":
        pi0_model = pi0_model.to(torch.bfloat16)
    else:
        raise ValueError(f"无效的精度: {precision}")

    # 使用safetensors保存转换后的模型
    os.makedirs(output_path, exist_ok=True)

    # 使用save_model保存模型权重为SafeTensors格式，处理绑定权重
    safetensors.torch.save_model(pi0_model, os.path.join(output_path, "model.safetensors"))

    # 如果存在assets文件夹则复制
    assets_source = pathlib.Path(checkpoint_dir).parent / "assets"
    if assets_source.exists():
        assets_dest = pathlib.Path(output_path) / "assets"
        if assets_dest.exists():
            shutil.rmtree(assets_dest)
        shutil.copytree(assets_source, assets_dest)

    # 保存配置为JSON以供参考
    config_dict = {
        "action_dim": model_config.action_dim,          # 动作维度
        "action_horizon": model_config.action_horizon,  # 动作时间范围
        "paligemma_variant": model_config.paligemma_variant,      # PaliGemma变体
        "action_expert_variant": model_config.action_expert_variant,  # 动作专家变体
        "precision": precision,                         # 精度
    }
    
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    print("模型转换成功完成！")
    print(f"模型已保存到 {output_path}")


def main(
    checkpoint_dir: str,
    config_name: str,
    output_path: str | None = None,
    precision: Literal["float32", "bfloat16", "float16"] = "bfloat16",
    *,
    inspect_only: bool = False,
):
    """加载JAX模型并可选择转换为PyTorch格式。

    Args:
        checkpoint_dir: JAX检查点目录路径
        config_name: 配置名称
        output_path: 保存转换后PyTorch模型的路径（转换时必需）
        precision: 模型转换精度
        inspect_only: 仅检查参数键，不进行转换
    """
    # 获取模型配置
    model_config = _config.get_config(config_name).model
    
    # 检查配置是否为Pi0Config类型
    if not isinstance(model_config, openpi.models.pi0_config.Pi0Config):
        raise ValueError(f"配置 {config_name} 不是 Pi0Config 类型")
    
    # 根据参数决定执行检查还是转换
    if inspect_only:
        load_jax_model_and_print_keys(checkpoint_dir)
    else:
        if not output_path:
            print("错误: 转换时 --output_path 是必需的。使用 --inspect_only 仅查看键。")
            return
        convert_pi0_checkpoint(checkpoint_dir, precision, output_path, model_config)


if __name__ == "__main__":
    tyro.cli(main)