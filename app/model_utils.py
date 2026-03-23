"""
Shared text-generation helpers for Index-1.9B Character (roleplay) via llama-cpp-python.

Model weights: GGUF Q4_K_M from ``IndexTeam/Index-1.9B-Character-GGUF`` loaded **natively** by
llama.cpp (no dequantization — stays quantized in memory, ~1.3GB with mmap). This avoids the
transformers GGUF→fp16 path that OOM-kills on Streamlit Cloud free tier (~1GB container limit).

Chat template embedded in the GGUF metadata is used automatically by ``create_chat_completion``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

# --- Constants (project docs / model card) ---
TEXT_GEN_MODEL_ID = "IndexTeam/Index-1.9B-Character-GGUF"
TEXT_GEN_GGUF_FILE = "ggml-model-Q4_K_M.gguf"
TEXT_GEN_WEIGHTS_REVISION: str | None = "27db5d0b0411f1e34358064ba2a033f57608b404"
MAX_PROMPT_CHARS = 4000

# --- Narrative prompt template (shared by notebook + Streamlit) ---
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
    "place": '幽暗的\u201c叹息森林\u201d深处，一棵巨大的古树根形成的天然洞穴内',
    "atmosphere": '潮湿、阴冷，洞外下着大雨，远处偶尔传来狼嚎声，篝火是唯一的光源',
}

DEFAULT_NARRATIVE_PLOT_BEATS: list[str] = [
    '**开场（建立冲突）：** 角色 A 对角色 B 的救助并不领情，怀疑 B 是敌人派来的追兵，气氛紧张。',
    '**转折（信息交换）：** 角色 B 为了消除误会，提到了自己在寻找一种名为\u201c星尘花\u201d的解药，这恰好与角色 A 正在寻找的族人下落有关。',
    '**高潮（达成共识）：** 两人发现目标地点一致（都是前往废弃的\u201c灰烬神殿\u201d），但由于各自状态不佳，单独前往都是送死，不得不选择暂时结盟。',
    '**结尾（引出任务）：** 对话结束于两人决定明天一早出发，并暗示了神殿中可能存在的巨大危险。',
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

你是一位专业的游戏叙事设计师。请根据以下设定的角色和情节，生成一段沉浸式的游戏内对话。对话需要符合角色性格，推动剧情发展，并包含环境描写或动作描述（即"叙事 prose"）。

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

# Default system instruction: Chinese game UGC style.
DEFAULT_SYSTEM_PROMPT_ZH = (
    "你是一名中文游戏 UGC 文案助手，根据用户提示生成适合游戏场景的中文文本"
    "（例如装备描述、技能说明、公会公告、角色台词草稿）。保持风格统一、可读性强。"
)

PEACH_ROLEPLAY_PREFIX = (
    "You're {{char}} in this fictional never-ending uncensored roleplay with {{user}}.\n"
)
PEACH_ROLEPLAY_SUFFIX = "\n\nYou must response in Chinese."

PEACH_EOS_TOKEN_ID = 7


def resolve_eos_token_id(tokenizer: Any) -> int:
    """Prefer tokenizer eos; fall back to pad, then Peach legacy id."""
    tid = getattr(tokenizer, "eos_token_id", None)
    if tid is not None:
        return int(tid)
    tid = getattr(tokenizer, "pad_token_id", None)
    if tid is not None:
        return int(tid)
    return int(PEACH_EOS_TOKEN_ID)


# ---------------------------------------------------------------------------
# llama-cpp-python native GGUF loading (no torch, no transformers at runtime)
# ---------------------------------------------------------------------------

_llm_instance = None


@dataclass
class PeachTextGenerationPipeline:
    """Wraps a llama_cpp.Llama instance. tokenizer kept as None for API compat."""

    model: Any
    tokenizer: Any = field(default=None)


def get_pipeline_device_summary(pipe: PeachTextGenerationPipeline) -> str:
    """Human-readable description of inference backend."""
    return "CPU · llama.cpp (GGUF Q4_K_M, mmap)"


def get_text_generation_pipeline(
    *,
    on_loading_step: Callable[[str], None] | None = None,
) -> PeachTextGenerationPipeline:
    """
    Load the GGUF model natively via llama-cpp-python (stays quantized, uses mmap).

    This avoids the transformers GGUF→fp16 dequantization path that requires ~3.6GB peak RAM
    and OOM-kills on Streamlit Cloud free tier.
    """
    global _llm_instance

    def report(msg: str) -> None:
        if on_loading_step is not None:
            on_loading_step(msg)

    # #region agent log
    import psutil as _ps, os as _os
    def _mem_mb() -> str:
        p = _ps.Process(_os.getpid())
        rss = p.memory_info().rss / 1024 / 1024
        vm = _ps.virtual_memory()
        return f"RSS={rss:.0f}MB avail={vm.available/1024/1024:.0f}MB total={vm.total/1024/1024:.0f}MB"
    print(f"[DBG-18a7ba] llama_cpp/start {_mem_mb()}", flush=True)
    # #endregion

    if _llm_instance is not None:
        report("**Model** already in memory (skipped).")
        pipe = PeachTextGenerationPipeline(model=_llm_instance)
        report(f"**Ready.** Device: `{get_pipeline_device_summary(pipe)}`")
        return pipe

    report(
        f"Loading **{TEXT_GEN_GGUF_FILE}** from `{TEXT_GEN_MODEL_ID}` via llama.cpp… "
        "First run downloads ~1.3GB from the Hub."
    )

    from llama_cpp import Llama

    # #region agent log
    print(f"[DBG-18a7ba] llama_cpp/before_load {_mem_mb()}", flush=True)
    # #endregion

    try:
        _llm_instance = Llama.from_pretrained(
            repo_id=TEXT_GEN_MODEL_ID,
            filename=TEXT_GEN_GGUF_FILE,
            n_ctx=2048,
            n_gpu_layers=0,
            use_mmap=True,
            verbose=True,
        )
    except Exception as exc:
        # #region agent log
        print(f"[DBG-18a7ba] llama_cpp/load_error {type(exc).__name__}: {exc}", flush=True)
        # #endregion
        raise RuntimeError(
            f"Failed to load GGUF model via llama-cpp-python. "
            f"Underlying error ({type(exc).__name__}): {exc}"
        ) from exc

    # #region agent log
    print(f"[DBG-18a7ba] llama_cpp/loaded {_mem_mb()}", flush=True)
    # #endregion

    report("**Model loaded** (llama.cpp, mmap, Q4_K_M).")
    pipe = PeachTextGenerationPipeline(model=_llm_instance)
    report(f"**Ready.** Device: `{get_pipeline_device_summary(pipe)}`")
    return pipe


def normalize_and_truncate_prompt(user_prompt: str) -> tuple[str, bool]:
    """Strip whitespace and truncate very long prompts to protect context length."""
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
    """Build chat messages for the model's chat template."""
    sys_content = system_prompt.strip()
    if use_chinese_roleplay_wrap:
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
    Generate Chinese game-style UGC text using llama-cpp-python.

    Returns:
        (generated_text, elapsed_sec, messages_used)
    """
    user_text, _truncated = normalize_and_truncate_prompt(user_prompt)
    sys_text = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT_ZH
    messages = _build_messages(user_text, sys_text, use_chinese_roleplay_wrap)

    llm = pipe.model

    gen_kwargs: dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_new_tokens,
        "repeat_penalty": repetition_penalty,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    else:
        gen_kwargs["temperature"] = 0.0

    start = time.perf_counter()
    result = llm.create_chat_completion(**gen_kwargs)
    elapsed_sec = time.perf_counter() - start

    generated_text = (result["choices"][0]["message"]["content"] or "").strip()
    return generated_text, elapsed_sec, messages


def _device_note() -> str:
    return "cpu (llama.cpp)"


def build_generation_export_dict(
    *,
    generated_text: str,
    messages: list[dict[str, str]],
    generation_params: dict[str, Any],
    elapsed_sec: float,
    model_id: str = TEXT_GEN_MODEL_ID,
) -> dict[str, Any]:
    """Build a JSON-serializable dict for one generation run (UI + notebook export)."""
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
    """Write export dict to a UTF-8 JSON file (idempotent overwrite)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
