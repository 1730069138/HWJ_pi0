import dataclasses
import numpy as np
import einops
from openpi import transforms
from openpi.models import model as _model

def _parse_image(image) -> np.ndarray:
    """将图像解析为模型所需的 uint8 (H,W,C) 格式"""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image

@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # 解析图像。假设我们将 global（全局俯视）作为主视角，wrist（手腕）作为腕部视角
        # 你也可以把 cam_side 作为主视角，这里以 global 为例
        cam_global = _parse_image(data["observation/cam_global"])
        cam_wrist = _parse_image(data["observation/cam_wrist"])

        # 构建输入字典
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": cam_global,          # Pi0模型要求的主视角名称
                "left_wrist_0_rgb": cam_wrist,     # Pi0模型要求的腕部视角名称
                "right_wrist_0_rgb": np.zeros_like(cam_global), # 无右手腕部视角，用0填充
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # 填充 actions 和 prompt（指令）
        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs

@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # 推理时截取前 N 个动作。Piper 机械臂通常是 6 个关节 + 1 个夹爪 = 7 个维度
        return {"actions": np.asarray(data["actions"][:, :7])}