# 敏感词表（在线拉取，不落盘）

应用 **不再** 从本目录读取 `wordlist.txt`。词表在运行时通过 HTTPS 从上游仓库拉取，**仅在进程内存中缓存**（见 [`app/toxic_classification_pipeline.py`](../../toxic_classification_pipeline.py) 中的 `TENCENT_SENSITIVE_WORDS_URL` 与 `load_sensitive_words()`）。

## 上游来源

- 仓库：[cjh0613/tencent-sensitive-words](https://github.com/cjh0613/tencent-sensitive-words)
- 当前使用的 raw 文件：`sensitive_words_lines.txt`（`main` 分支，一行一词）
- 许可证：请阅读上游仓库 **LICENSE**；本项目不随仓库分发词表正文。

## 部署要求

- **首次** 执行 Step 2 检测前需要能访问 `raw.githubusercontent.com`（与拉取 HF 模型类似）。
- 若下载失败，界面会报错；请检查网络、防火墙或代理。

本目录仅保留说明文档；无需向此处拷贝词表文件。
