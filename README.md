# Game UGC Generation & Safety Review App

This folder contains a Streamlit application for Chinese game UGC workflows:

1. Generate game-style text with a Hugging Face text-generation model.
2. Run sentence-level safety checks before publication.
3. Save publication history and export compliance-only JSON records.

## Overview

The app is a two-step moderation workflow:

- **Step 1: Generation**
  - Generate Chinese UGC text from either:
    - a structured narrative template, or
    - a free-form prompt.
  - Supports decoding controls (`do_sample`, `temperature`, `top_p`, `max_new_tokens`, etc.).

- **Step 2: Safety review**
  - Splits generated text into sentence-like units.
  - Reviews each sentence with:
    - a fine-tuned toxic classifier model on Hugging Face, and
    - a Tencent-style sensitive-word list downloaded from GitHub.
  - Publishing is enabled only when all sentences are checked and pass.

## Main Files

- `app.py`  
  Streamlit UI and end-to-end workflow (generation, review, publishing, history).

- `model_utils.py`  
  Text-generation model loading, prompt building, prompt normalization/truncation, and generation helpers.

- `toxic_classification_pipeline.py`  
  Toxic classification pipeline, sensitive-wordlist fetching/scanning, sentence splitting, and safety decision logic.

- `requirements.txt`  
  Runtime dependencies for Streamlit + Transformers + PyTorch stack.

## Models and External Data

- **Text generation model**
  - `Qwen/Qwen2.5-0.5B-Instruct`
  - Revision pinned in code:
    - `7ae557604adf67be50417f59c2c2f167def9a775`

- **Toxic classifier model**
  - `echovivi/CustomModel_bertToxiCN`

- **Sensitive word list source**
  - [cjh0613/tencent-sensitive-words](https://github.com/cjh0613/tencent-sensitive-words)
  - Raw list URL used in code:
    - `https://raw.githubusercontent.com/cjh0613/tencent-sensitive-words/main/sensitive_words_lines.txt`

## Installation

From the repository root:

```bash
pip install -r app/requirements.txt
```

## Run Locally

From the repository root:

```bash
streamlit run app/app.py
```

Then open the local Streamlit URL shown in terminal.

## How the Safety Gate Works

For each sentence:

- Run classifier prediction (`toxic` / `non_toxic`).
- Scan sentence for sensitive word hits.
- Mark sentence as unsafe if **either** check fails.

Publishing is allowed only when:

- every sentence is non-empty,
- every sentence has been checked,
- no sentence is marked unsafe.

## Publication History and Compliance Export

Published records are stored at:

- `app/data/publish_history.json`

Each record includes a compliance payload (no raw moderation internals required for publication export), such as:

- schema version
- status
- entry id
- published timestamp
- classifier model id
- full text
- sentence list

The UI supports:

- viewing read-only history
- downloading per-record compliance JSON
- downloading a full compliance bundle JSON

## Notes

- First run may be slow due to model downloads from Hugging Face.
- In hosted environments (for example Streamlit Community Cloud), local file history may be ephemeral depending on deployment settings.
- The app uses in-process caching for model pipelines to reduce repeated load time.
