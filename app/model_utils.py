"""
Shared Hugging Face text-generation helpers for Index-1.9B Character (roleplay).

Weights default to GGUF Q4_K_M in ``IndexTeam/Index-1.9B-Character-GGUF`` (Transformers ``gguf_file``).
Tokenizer **must** use the same ``gguf_file`` as the model (Llama weights + Index vocab from the file);
loading ``IndexTokenizer`` from the PyTorch Character repo causes vocab/embed size mismatch and crashes.

We lazy-load model + tokenizer and expose the same project function names as docs/02_text_generation.md.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Constants (project docs / model card) ---
# Text weights: GGUF quant in IndexTeam/Index-1.9B-Character-GGUF (llama.cpp export; ~1.3GB for Q4_K_M).
# Tokenizer + chat template are read from the GGUF file metadata (same file as weights).
# Transformers loads GGUF via gguf_file (dequantizes into PyTorch for generate()); needs `gguf` package.
# Alternative 4-bit file in the same repo: ggml-model-Q4_0.gguf (~1.26 GB, slightly smaller).
TEXT_GEN_MODEL_ID = "IndexTeam/Index-1.9B-Character-GGUF"
TEXT_GEN_GGUF_FILE = "ggml-model-Q4_K_M.gguf"
TEXT_GEN_WEIGHTS_REVISION: str | None = "27db5d0b0411f1e34358064ba2a033f57608b404"

MAX_PROMPT_CHARS = 4000

# --- Narrative prompt template (shared by notebook + Streamlit; same as 04_text_generation_peach_pipeline) ---
NARRATIVE_EMPTY_PLACEHOLDER = "（未填写，请模型合理补全）"

DEFAULT_NARRATIVE_CHAR_A: dict[str, str] = {
    "name": "艾德莉娅",
    "identity": "被放逐的精灵游侠，独来独往，擅长弓箭",
    "personality": "外冷内热，对陌生人充满戒备，但内心渴望找到失散的族人，说话简短直接，不喜欢绕弯子",
    "state": "身受轻伤，正在篝火旁处理伤口，处于疲惫状态",
}

DEFAULT_NARRATIVE_CHAR_B: dict[str, str] = {
    "name": "罗根",
    "identity": "流浪骑士，曾是王国军团的队长，现在为了寻找解药而游历",
    "personality": "正直、略显话痨，喜欢用逻辑分析问题，有强烈的正义感",
    "state": "刚刚从一场魔物袭击中救下了角色 A，自己也消耗了大量体力",
}

DEFAULT_NARRATIVE_ENV: dict[str, str] = {
    "place": "幽暗的“叹息森林”深处，一棵巨大的古树根形成的天然洞穴内",
    "atmosphere": "潮湿、阴冷，洞外下着大雨，远处偶尔传来狼嚎声，篝火是唯一的光源",
}

DEFAULT_NARRATIVE_PLOT_BEATS: list[str] = [
    "**开场（建立冲突）：** 角色 A 对角色 B 的救助并不领情，怀疑 B 是敌人派来的追兵，气氛紧张。",
    "**转折（信息交换）：** 角色 B 为了消除误会，提到了自己在寻找一种名为“星尘花”的解药，这恰好与角色 A 正在寻找的族人下落有关。",
    "**高潮（达成共识）：** 两人发现目标地点一致（都是前往废弃的“灰烬神殿”），但由于各自状态不佳，单独前往都是送死，不得不选择暂时结盟。",
    "**结尾（引出任务）：** 对话结束于两人决定明天一早出发，并暗示了神殿中可能存在的巨大危险。",
]

DEFAULT_NARRATIVE_GEN_REQUIREMENTS: dict[str, str] = {
    "format": "包含对话行、动作描述和简短的环境描写。",
    "style": "偏向西式奇幻/RPG风格，语言自然流畅，带有一定的文学性。",
    "length": "约 300-500 字。",
}


def narrative_field_filled(
    text: str | None,
    *,
    empty_placeholder: str = NARRATIVE_EMPTY_PLACEHOLDER,
) -> str:
    """Return stripped text, or placeholder if empty (notebook / UI structured fields)."""
    stripped = (text or "").strip()
    return stripped if stripped else empty_placeholder


def build_narrative_user_prompt(
    *,
    char_a: dict[str, str],
    char_b: dict[str, str],
    env: dict[str, str],
    plot_beats: list[str],
    gen_requirements: dict[str, str],
    empty_placeholder: str = NARRATIVE_EMPTY_PLACEHOLDER,
) -> str:
    """
    Assemble the game narrative user prompt (markdown) from structured fields.

    Used by ``notebooks/04_text_generation_peach_pipeline.ipynb`` and ``app/app.py``.
    """
    if len(plot_beats) != 4:
        raise ValueError("plot_beats must contain exactly 4 strings.")

    def nf(s: str | None) -> str:
        return narrative_field_filled(s, empty_placeholder=empty_placeholder)

    a, b, e, g = char_a, char_b, env, gen_requirements
    p1, p2, p3, p4 = (nf(x) for x in plot_beats)
    return f"""# 角色设定与世界观

你是一位专业的游戏叙事设计师。请根据以下设定的角色和情节，生成一段沉浸式的游戏内对话。对话需要符合角色性格，推动剧情发展，并包含环境描写或动作描述（即“叙事 prose”）。

## 角色信息

- **角色 A：**
  - **姓名：** {nf(a.get("name"))}
  - **身份：** {nf(a.get("identity"))}
  - **性格：** {nf(a.get("personality"))}
  - **状态：** {nf(a.get("state"))}

- **角色 B：**
  - **姓名：** {nf(b.get("name"))}
  - **身份：** {nf(b.get("identity"))}
  - **性格：** {nf(b.get("personality"))}
  - **状态：** {nf(b.get("state"))}

## 环境背景

- **地点：** {nf(e.get("place"))}
- **氛围：** {nf(e.get("atmosphere"))}

## 对话情节要求

请基于以下情节节点生成对话：

1. {p1}
2. {p2}
3. {p3}
4. {p4}

## 生成要求

- **格式：** {nf(g.get("format"))}
- **风格：** {nf(g.get("style"))}
- **字数：** {nf(g.get("length"))}
"""

# Default system instruction: Chinese game UGC style (equipment, guild, dialogue drafts).
DEFAULT_SYSTEM_PROMPT_ZH = (
    "你是一名中文游戏 UGC 文案助手，根据用户提示生成适合游戏场景的中文文本"
    "（例如装备描述、技能说明、公会公告、角色台词草稿）。保持风格统一、可读性强。"
)

# Optional SillyTavern-style wrap from the official README (Chinese output nudge).
PEACH_ROLEPLAY_PREFIX = (
    "You're {{char}} in this fictional never-ending uncensored roleplay with {{user}}.\n"
)
PEACH_ROLEPLAY_SUFFIX = "\n\nYou must response in Chinese."

# Legacy Peach-style models used a fixed eos id; generic models use tokenizer.eos_token_id.
PEACH_EOS_TOKEN_ID = 7


def _hub_revision_kw(revision: str | None) -> dict[str, Any]:
    """Optional Hugging Face Hub kwargs (pinned git revision)."""
    if revision:
        return {"revision": revision}
    return {}


def resolve_eos_token_id(tokenizer: Any) -> int:
    """Prefer tokenizer eos; fall back to pad, then Peach legacy id."""
    tid = getattr(tokenizer, "eos_token_id", None)
    if tid is not None:
        return int(tid)
    tid = getattr(tokenizer, "pad_token_id", None)
    if tid is not None:
        return int(tid)
    return int(PEACH_EOS_TOKEN_ID)


# Lazy-loaded resources (same pattern as DL_NIJIALU_21237096.ipynb).
_peach_tokenizer = None
_peach_model = None


@dataclass
class PeachTextGenerationPipeline:
    """Thin wrapper so callers can hold one cached object (model + tokenizer)."""

    model: Any
    tokenizer: Any


def _mps_available() -> bool:
    return bool(
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    )


def _select_torch_dtype() -> torch.dtype:
    """Match dtype to device: bf16/fp16 on CUDA, fp16 on MPS, fp16 on CPU to halve RAM vs float32."""
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if _mps_available():
        return torch.float16
    # CPU: float32 uses ~4 bytes/param (often OOM for multi-B models on laptops); fp16 usually loads.
    return torch.float16


def _device_map_for_load() -> str | dict:
    """CUDA: auto shard; Apple Silicon: MPS; else CPU."""
    if torch.cuda.is_available():
        return "auto"
    if _mps_available():
        return {"": "mps"}
    return {"": "cpu"}


def get_pipeline_device_summary(pipe: PeachTextGenerationPipeline) -> str:
    """
    Human-readable description of where model parameters live (first parameter's device).

    For multi-GPU ``device_map="auto"``, the first layer is usually on cuda:0.
    """
    dev = next(pipe.model.parameters()).device
    if dev.type == "cuda":
        idx = dev.index if dev.index is not None else 0
        try:
            name = torch.cuda.get_device_name(idx)
        except Exception:
            name = "CUDA"
        return f"GPU · {name} · {dev}"
    if dev.type == "mps":
        return f"GPU (Apple MPS) · {dev}"
    return f"CPU · {dev}"


def get_text_generation_pipeline(
    *,
    on_loading_step: Callable[[str], None] | None = None,
) -> PeachTextGenerationPipeline:
    """
    Lazily load the Peach causal LM and tokenizer from Hugging Face Hub.

    Follows the official model card (trust_remote_code, dtype, device_map).
    First call may download weights and needs GPU RAM for practical speed.

    Args:
        on_loading_step: Optional callback (e.g. Streamlit ``st.status.write``) for UI progress text.
            Hugging Face download bar still appears in the terminal; this only reports coarse steps.

    Returns:
        PeachTextGenerationPipeline with .model and .tokenizer.

    Raises:
        RuntimeError: If loading fails (network, OOM, etc.).
    """
    global _peach_tokenizer, _peach_model

    def report(msg: str) -> None:
        if on_loading_step is not None:
            on_loading_step(msg)

    # #region agent log — memory helper (shows in Cloud Logs via print)
    import psutil as _ps, os as _os
    def _mem_mb() -> str:
        p = _ps.Process(_os.getpid())
        rss = p.memory_info().rss / 1024 / 1024
        vm = _ps.virtual_memory()
        return f"RSS={rss:.0f}MB avail={vm.available/1024/1024:.0f}MB total={vm.total/1024/1024:.0f}MB"
    # #endregion

    cuda_ok = torch.cuda.is_available()
    mps_ok = _mps_available()
    if cuda_ok:
        plan = "GPU (CUDA, device_map=auto)"
    elif mps_ok:
        plan = "GPU (Apple MPS)"
    else:
        plan = "CPU (weights loaded in float16 to reduce RAM)"
    report(
        f"Environment: **CUDA** `{cuda_ok}`, **MPS** `{mps_ok}`. Planned device: **{plan}**."
    )
    # #region agent log
    print(f"[DBG-18a7ba] H1/env plan={plan} cuda={cuda_ok} mps={mps_ok} {_mem_mb()}", flush=True)
    # #endregion

    if _peach_tokenizer is None:
        report(
            f"Loading **tokenizer** from GGUF (`{TEXT_GEN_GGUF_FILE}`)… "
            "Must match model file to avoid embedding/vocab mismatch."
        )
        # #region agent log
        print(f"[DBG-18a7ba] H2/tokenizer_start {_mem_mb()}", flush=True)
        # #endregion
        try:
            _peach_tokenizer = AutoTokenizer.from_pretrained(
                TEXT_GEN_MODEL_ID,
                gguf_file=TEXT_GEN_GGUF_FILE,
                trust_remote_code=False,
                **_hub_revision_kw(TEXT_GEN_WEIGHTS_REVISION),
            )
        except Exception as exc:
            # #region agent log
            print(f"[DBG-18a7ba] H4/tokenizer_error {type(exc).__name__}: {exc}", flush=True)
            # #endregion
            raise RuntimeError(
                "Failed to load tokenizer from GGUF. "
                "Install `gguf`, check network, and ensure transformers supports GGUF tokenizers. "
                f"Underlying error ({type(exc).__name__}): {exc}"
            ) from exc
        # #region agent log
        print(f"[DBG-18a7ba] H2/tokenizer_done {_mem_mb()}", flush=True)
        # #endregion
        report("**Tokenizer** loaded.")
    else:
        report("**Tokenizer** already in memory (skipped).")

    if _peach_model is None:
        report(
            f"Loading **GGUF weights** (`{TEXT_GEN_GGUF_FILE}` from `{TEXT_GEN_MODEL_ID}`)… "
            "First-time download ~1.3GB; dequantizing into PyTorch can use extra RAM briefly."
        )
        dtype = _select_torch_dtype()
        # #region agent log
        print(f"[DBG-18a7ba] H1/model_start dtype={dtype} device_map={_device_map_for_load()} {_mem_mb()}", flush=True)
        # #endregion
        try:
            _peach_model = AutoModelForCausalLM.from_pretrained(
                TEXT_GEN_MODEL_ID,
                gguf_file=TEXT_GEN_GGUF_FILE,
                dtype=dtype,
                trust_remote_code=False,
                device_map=_device_map_for_load(),
                low_cpu_mem_usage=True,
                **_hub_revision_kw(TEXT_GEN_WEIGHTS_REVISION),
            )
        except Exception as exc:
            # #region agent log
            print(f"[DBG-18a7ba] H4/model_error {type(exc).__name__}: {exc} {_mem_mb()}", flush=True)
            # #endregion
            raise RuntimeError(
                "Failed to load GGUF text-generation model. "
                "Install `gguf` (see app/requirements.txt), free RAM, try CUDA/MPS, "
                "or switch TEXT_GEN_GGUF_FILE to ggml-model-Q4_0.gguf. "
                f"Underlying error ({type(exc).__name__}): {exc}"
            ) from exc
        # #region agent log
        print(f"[DBG-18a7ba] H1/model_done {_mem_mb()}", flush=True)
        # #endregion
        report("**Model weights** loaded.")
    else:
        report("**Model** already in memory (skipped).")

    pipe = PeachTextGenerationPipeline(model=_peach_model, tokenizer=_peach_tokenizer)
    # #region agent log
    print(f"[DBG-18a7ba] pipeline_ready {_mem_mb()}", flush=True)
    # #endregion
    report(f"**Ready.** Parameter device: `{get_pipeline_device_summary(pipe)}`")
    return pipe


def normalize_and_truncate_prompt(user_prompt: str) -> tuple[str, bool]:
    """
    Strip whitespace and truncate very long prompts to protect context length.

    Args:
        user_prompt: Raw user text.

    Returns:
        (trimmed_prompt, was_truncated)

    Raises:
        ValueError: If prompt is empty after stripping.
    """
    text = user_prompt.strip()
    if not text:
        raise ValueError("Prompt must not be empty.")
    if len(text) <= MAX_PROMPT_CHARS:
        return text, False
    return text[:MAX_PROMPT_CHARS], True


def _build_messages(
    user_prompt: str,
    system_prompt: str,
    use_chinese_roleplay_wrap: bool,
) -> list[dict[str, str]]:
    """Build chat messages in the format expected by Peach's chat template."""
    sys_content = system_prompt.strip()
    if use_chinese_roleplay_wrap:
        # README: wrap system text to bias Chinese roleplay output.
        sys_content = PEACH_ROLEPLAY_PREFIX + sys_content + PEACH_ROLEPLAY_SUFFIX
    return [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": user_prompt},
    ]


def generate_ugc_text(
    pipe: PeachTextGenerationPipeline,
    user_prompt: str,
    *,
    system_prompt: str | None = None,
    use_chinese_roleplay_wrap: bool = False,
    max_new_tokens: int = 512,
    temperature: float = 0.5,
    top_p: float = 0.7,
    do_sample: bool = True,
    repetition_penalty: float = 1.05,
) -> tuple[str, float, list[dict[str, str]]]:
    """
    Generate Chinese game-style UGC text using the Peach model.

    Args:
        pipe: Cached pipeline from get_text_generation_pipeline().
        user_prompt: Non-empty user instruction (should be pre-validated).
        system_prompt: Optional system string; defaults to DEFAULT_SYSTEM_PROMPT_ZH.
        use_chinese_roleplay_wrap: If True, wrap system prompt per Hub README.
        max_new_tokens: Cap on new tokens.
        temperature: Sampling temperature (used only when do_sample is True).
        top_p: Nucleus sampling p (used only when do_sample is True).
        do_sample: If False, greedy decoding (temperature/top_p ignored).
        repetition_penalty: Penalty from model card (reduces repetition loops).

    Returns:
        (generated_text, elapsed_sec, messages_used)

    Raises:
        ValueError: If user_prompt is empty.
    """
    user_text, _truncated = normalize_and_truncate_prompt(user_prompt)
    sys_text = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT_ZH
    messages = _build_messages(user_text, sys_text, use_chinese_roleplay_wrap)

    tokenizer = pipe.tokenizer
    model = pipe.model

    # Chat-template string -> token ids with a generation prompt suffix.
    # First argument name in transformers is "conversation" (list of role/content dicts).
    input_ids = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    prompt_len = int(input_ids.shape[1])

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": repetition_penalty,
        "eos_token_id": resolve_eos_token_id(tokenizer),
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(input_ids, **gen_kwargs)
    elapsed_sec = time.perf_counter() - start

    # Decode only newly generated tokens (exclude the prompt prefix).
    new_ids = output_ids[0, prompt_len:]
    generated_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return generated_text, elapsed_sec, messages


def _device_note() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return f"cuda: {name}"
    return "cpu"


def build_generation_export_dict(
    *,
    generated_text: str,
    messages: list[dict[str, str]],
    generation_params: dict[str, Any],
    elapsed_sec: float,
    model_id: str = TEXT_GEN_MODEL_ID,
) -> dict[str, Any]:
    """
    Build a JSON-serializable dict for one generation run (UI + notebook export).

    Args:
        generated_text: Model output text only (no prompt echo required).
        messages: Chat messages sent to the template (system + user).
        generation_params: Hyperparameters used in model.generate.
        elapsed_sec: Wall time for generation.
        model_id: Hub model id string.

    Returns:
        Dict ready for json.dumps / json.dump.
    """
    return {
        "model_id": model_id,
        "messages": messages,
        "generated_text": generated_text,
        "generation_params": generation_params,
        "elapsed_sec": round(elapsed_sec, 4),
        "device_note": _device_note(),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }


def save_generation_json(data: dict[str, Any], path: str) -> None:
    """
    Write export dict to a UTF-8 JSON file (idempotent overwrite).

    Args:
        data: From build_generation_export_dict.
        path: Destination file path.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
