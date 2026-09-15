import dataclasses
import numpy as np
import einops
from openpi import transforms
from openpi.models import model as _model

def _parse_image(image):
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image

@dataclasses.dataclass(frozen=True)
class DummyRealInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # 解析你实际录制的两个相机
        base_img = _parse_image(data["cam_global"])
        wrist_img = _parse_image(data["cam_wrist"])
        
        # 🚨 生成一张和全局相机一样大小的“全黑假照片”
        dummy_img = np.zeros_like(base_img)

        inputs = {
            "state": data["state"],
            "image": {
                "base_0_rgb": base_img,
                "left_wrist_0_rgb": wrist_img,
                "right_wrist_0_rgb": dummy_img,  # 塞入假照片骗过底层网络结构
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,  # 🚨 极其关键：告诉大模型彻底无视这张图！
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs

@dataclasses.dataclass(frozen=True)
class DummyRealOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # 6个关节 + 1个夹爪 = 7维动作
        return {"actions": np.asarray(data["actions"][:, :7])}