### Docker 设置

本仓库中的所有示例都提供了正常运行和使用 Docker 运行的说明。虽然不是必需的，但推荐使用 Docker 选项，因为这将简化软件安装，产生更稳定的环境，并且还可以让您避免安装 ROS 并弄乱您的机器（对于依赖 ROS 的示例）。

- 基本的 Docker 安装说明在[这里](https://docs.docker.com/engine/install/)。
- Docker 必须安装在[rootless 模式](https://docs.docker.com/engine/security/rootless/)下。
- 要使用您的 GPU，您还必须安装[NVIDIA 容器工具包](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。
- 使用 `snap` 安装的 docker 版本与 NVIDIA 容器工具包不兼容，导致其无法访问 `libnvidia-ml.so`（[问题](https://github.com/NVIDIA/nvidia-container-toolkit/issues/154)）。可以通过 `sudo snap remove docker` 卸载 snap 版本。
- Docker Desktop 也与 NVIDIA 运行时不兼容（[问题](https://github.com/NVIDIA/nvidia-container-toolkit/issues/229)）。可以通过 `sudo apt remove docker-desktop` 卸载 Docker Desktop。

如果从头开始并且您的主机是 Ubuntu 22.04，您可以使用便捷脚本 `scripts/docker/install_docker_ubuntu22.sh` 和 `scripts/docker/install_nvidia_container_toolkit.sh` 来完成以上所有操作。

使用以下命令构建 Docker 镜像并启动容器：
```bash
docker compose -f scripts/docker/compose.yml up --build
```

要为特定示例构建和运行 Docker 镜像，请使用以下命令：
```bash
docker compose -f examples/<example_name>/compose.yml up --build
```
其中 `<example_name>` 是您想要运行的示例名称。

在首次运行任何示例时，Docker 将构建镜像。去喝杯咖啡等待这个过程完成。后续运行会更快，因为镜像是缓存的。