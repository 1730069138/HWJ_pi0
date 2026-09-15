
# 远程运行 openpi 模型

我们提供了远程运行 openpi 模型的实用程序。这对于在机器人之外使用更强大的 GPU 运行推理非常有用，还有助于保持机器人和策略环境的分离（例如避免与机器人软件的依赖冲突）。

## 启动远程策略服务器

要启动远程策略服务器，您可以简单地运行以下命令：

```bash
uv run scripts/serve_policy.py --env=[DROID | ALOHA | LIBERO]
```

`env` 参数指定应加载哪个 $\pi_0$ 检查点。在底层，此脚本将执行类似以下的命令，您可以使用它来启动策略服务器，例如为您自己训练的检查点（这里是 DROID 环境的示例）：

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid --policy.dir=gs://openpi-assets/checkpoints/pi0_fast_droid
```

这将启动一个策略服务器，为 `config` 和 `dir` 参数指定的策略提供服务。该策略将在指定的端口（默认：8000）上提供服务。

## 从您的机器人代码查询远程策略服务器

我们提供了一个依赖最少的客户端实用程序，您可以轻松地将其嵌入到任何机器人代码库中。

首先，在您的机器人环境中安装 `openpi-client` 包：

```bash
cd $OPENPI_ROOT/packages/openpi-client
pip install -e .
```

然后，您可以使用客户端从您的机器人代码查询远程策略服务器。以下是执行此操作的示例：

```python
from openpi_client import image_tools
from openpi_client import websocket_client_policy

# 在回合循环外，初始化策略客户端。
# 指向策略服务器的主机和端口（localhost 和 8000 是默认值）。
client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

for step in range(num_steps):
    # 在回合循环内，构建观察。
    # 在客户端调整图像大小以最小化带宽/延迟。始终以 uint8 格式返回图像。
    # 我们提供用于调整图像大小和 uint8 转换的实用程序，以便您匹配训练例程。
    # 预训练 pi0 模型的典型 resize_size 是 224。
    # 注意，本体感知的 `state` 可以以未归一化的形式传递，归一化将在服务器端处理。
    observation = {
        "observation/image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        ),
        "observation/wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        ),
        "observation/state": state,
        "prompt": task_instruction,
    }

    # 使用当前观察调用策略服务器。
    # 这将返回一个形状为 (action_horizon, action_dim) 的动作块。
    # 注意，您通常只需要每隔 N 步调用一次策略，并在剩余步骤中开环执行预测的动作块中的步骤。
    action_chunk = client.infer(observation)["actions"]

    # 在环境中执行动作。
    ...

```

在这里，`host` 和 `port` 参数指定了远程策略服务器的 IP 地址和端口。您也可以将这些作为命令行参数指定给您的机器人代码，或在您的机器人代码库中硬编码。`observation` 是一个观察和提示字典，遵循您正在服务的策略的策略输入规范。我们在[简单客户端示例](../examples/simple_client/main.py)中提供了如何为不同环境构建此字典的具体示例。