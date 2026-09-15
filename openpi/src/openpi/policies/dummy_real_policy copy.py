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
        inputs = {
            "state": data["state"],
            "image": {
                # 必须使用模型底层硬编码的这三个键名！将实机的三个相机塞进去
                "base_0_rgb": _parse_image(data["cam_global"]),       # 全局相机当作 base
                "left_wrist_0_rgb": _parse_image(data["cam_wrist"]),  # 腕部相机当作 left_wrist
                "right_wrist_0_rgb": _parse_image(data["cam_side"]),  # 侧面相机当作 right_wrist
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
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