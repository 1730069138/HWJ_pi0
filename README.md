# HWJ_pi0

本仓库是本地 `openpi` 项目的私有备份仓库。

## 目录结构

```text
HWJ_pi0/
├── README.md
└── openpi/
    └── 项目源码及配置文件
```

项目主体位于 `openpi/` 目录中。

## 备份范围

仓库主要备份：

- OpenPI 项目源码
- 本地修改的训练与推理代码
- 配置文件和相关文档
- 小型配置备份压缩包

以下内容不会上传：

- 训练检查点和模型权重
- 训练集及演示数据
- 实验输出、日志和缓存
- Python 虚拟环境
- 包含密钥或令牌的环境配置
- 其他不适合存入 Git 的大型文件

这些内容仍保留在本地，不会因备份操作被删除。

## 更新备份

在本地 `openpi` 目录中完成修改后，依次运行：

```bash
git add -A
git commit -m "说明本次更新"
git hwj-push
```

由于远程仓库要求项目内容位于 `openpi/` 下，因此使用专用的：

```bash
git hwj-push
```

不要使用普通的 `git push`。

## Git 远程仓库

- `origin`：本私有备份仓库
- `upstream`：Physical Intelligence 官方 OpenPI 仓库

如需获取官方更新，可以从 `upstream` 拉取；本地修改和备份则推送至 `origin`。

## 注意事项

上传前应检查是否意外包含训练数据、模型权重、检查点或敏感信息。若新增了其他大型输出目录，应及时将其加入 `.gitignore`。
