# ISOM5240 Group Project — 中文游戏 UGC 生成与内容检测

（GitHub 仓库名：`deep_learning`）

## 文档

- 作业要求：[ISOM5240_project_requirements.pdf](ISOM5240_project_requirements.pdf)
- 代码规范：[执行指南_代码生成规范.md](执行指南_代码生成规范.md)
- **目录与 Canvas 提交映射**：[docs/00_项目文件架构与提交映射.md](docs/00_项目文件架构与提交映射.md)
- 功能流程：[docs/01_总功能流程.md](docs/01_总功能流程.md)

## 目录速览

| 路径 | 说明 |
|------|------|
| `notebooks/` | 微调：英文说明 [finetune_toxic_bert.ipynb](notebooks/finetune_toxic_bert.ipynb)；中文说明 [finetune_toxic_bert_zh.ipynb](notebooks/finetune_toxic_bert_zh.ipynb)（ToxiCN + `bert-base-chinese`） |
| `app/` | Streamlit：`app.py`、`requirements.txt`（见下 **Python / PyTorch**） |
| `data/` | 原始与处理后数据 |
| `models/finetuned/` | 与 Notebook 一致的微调权重 |
| `documentation/` | 提交用 `Project_report.pdf`、`Experimental_results.xlsx` |
| `presentation/` | PPT 与 `grpXX.mp4` |

填写 **Hugging Face Model URL**、**Streamlit Cloud App URL**、**GitHub URL** 到报告与本文档末尾（提交前）。

**Streamlit Cloud**：在应用的 **Settings → Advanced settings** 里把 **Python version** 选为 **3.12**（日志里若仍是 3.14，说明仅放 [`runtime.txt`](runtime.txt) 可能无效，以界面为准）。依赖见 [`app/requirements.txt`](app/requirements.txt)（含 **altair 4.x**、**gguf**）。文本生成默认使用 **`IndexTeam/Index-1.9B-Character-GGUF`** 的 **Q4_K_M**（约 **1.3GB** 下载）+ **`Index-1.9B-Character`** 的 tokenizer；首次点 **Generate** 仍可能较慢（下载 + 解压到 PyTorch），请到 **Manage app → Logs** 看进度。

## Python / PyTorch（本地跑 Streamlit）

- **推荐 Python 3.12 或 3.11**。默认文本生成权重为 **`IndexTeam/Index-1.9B-Character-GGUF`**（**Q4_K_M** GGUF，经 Transformers 载入），tokenizer 来自 **`IndexTeam/Index-1.9B-Character`**；仍需 **torch ≥ 2.6**（与 `app/requirements.txt` 一致）。
- **推代码到 Git 再在 Streamlit Cloud 部署**，往往能解决本机 **macOS Intel** 上 `pip` 装不到 **torch 2.6+** 的问题：云端一般是 **Linux**，PyTorch 对 **Linux x86_64** 的 **2.6+ wheel** 更全。注意 **Community Cloud 免费档内存/超时有限**，CPU 推理仍可能 **偏慢或超时**；不够用时需 **付费资源、GPU Space、或自建带 GPU/大内存的机器**。
- 若坚持在 **Intel Mac 本机**开发，请用 **conda** 等能装 **torch≥2.6** 的环境，或临时把 `model_utils` 里的模型换成更小的 Hub 模型做联调。
- 若使用 **Python 3.13**，尤其在 **macOS Intel (x86_64)** 上，`pip` 常会出现 **`No matching distribution found for torch`**，虚拟环境里也会 `ModuleNotFoundError: torch`（尚未装成功）。请改用 3.12/3.11 重建 venv，例如：
  - `brew install python@3.12`（或从 [python.org](https://www.python.org/downloads/) 安装 3.12），然后：
  - `cd <项目根目录> && rm -rf .venv && /usr/local/bin/python3.12 -m venv .venv`（路径按本机 `python3.12` 为准）
  - `source .venv/bin/activate && pip install -U pip && pip install -r app/requirements.txt`
  - `cd app && streamlit run app.py`
- 已用 **Anaconda** 时，也可用 `conda install pytorch`（选官方 channel 且版本 ≥2.6），再 `pip install` 其余 `requirements.txt` 中的包（避免重复装 torch 时可 `pip install` 时跳过 torch 行）。
