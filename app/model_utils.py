"""
Shared Hugging Face text-generation helpers for Qwen2.5-0.5B-Instruct.

Uses standard ``transformers`` AutoModelForCausalLM + AutoTokenizer (no trust_remote_code).
Safetensors format, ~1GB in float16. Chat template (ChatML ``<|im_start|>``) is built into
the tokenizer config; ``apply_chat_template`` works out of the box.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Constants ---
TEXT_GEN_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TEXT_GEN_MODEL_REVISION: str | None = "7ae557604adf67be50417f59c2c2f167def9a775"
MAX_PROMPT_CHARS = 4000

# --- Narrative prompt template (shared by notebook + Streamlit) ---
NARRATIVE_EMPTY_PLACEHOLDER = '\u201c\u672a\u586b\u5199\uff0c\u8bf7\u6a21\u578b\u5408\u7406\u8865\u5168\u201d'

DEFAULT_NARRATIVE_CHAR_A: dict[str, str] = {
    "name": "\u827e\u5fb7\u8389\u5a05",
    "identity": "\u88ab\u653e\u9010\u7684\u7cbe\u7075\u6e38\u4fa0\uff0c\u72ec\u6765\u72ec\u5f80\uff0c\u64c5\u957f\u5f13\u7bad",
    "personality": "\u5916\u51b7\u5185\u70ed\uff0c\u5bf9\u964c\u751f\u4eba\u5145\u6ee1\u6212\u5907\uff0c\u4f46\u5185\u5fc3\u6e34\u671b\u627e\u5230\u5931\u6563\u7684\u65cf\u4eba\uff0c\u8bf4\u8bdd\u7b80\u77ed\u76f4\u63a5\uff0c\u4e0d\u559c\u6b22\u7ed5\u5f2f\u5b50",
    "state": "\u8eab\u53d7\u8f7b\u4f24\uff0c\u6b63\u5728\u7bc5\u706b\u65c1\u5904\u7406\u4f24\u53e3\uff0c\u5904\u4e8e\u75b2\u60eb\u72b6\u6001",
}

DEFAULT_NARRATIVE_CHAR_B: dict[str, str] = {
    "name": "\u7f57\u6839",
    "identity": "\u6d41\u6d6a\u9a91\u58eb\uff0c\u66fe\u662f\u738b\u56fd\u519b\u56e2\u7684\u961f\u957f\uff0c\u73b0\u5728\u4e3a\u4e86\u5bfb\u627e\u89e3\u836f\u800c\u6e38\u5386",
    "personality": "\u6b63\u76f4\u3001\u7565\u663e\u8bdd\u75e8\uff0c\u559c\u6b22\u7528\u903b\u8f91\u5206\u6790\u95ee\u9898\uff0c\u6709\u5f3a\u70c8\u7684\u6b63\u4e49\u611f",
    "state": "\u521a\u521a\u4ece\u4e00\u573a\u9b54\u7269\u88ad\u51fb\u4e2d\u6551\u4e0b\u4e86\u89d2\u8272 A\uff0c\u81ea\u5df1\u4e5f\u6d88\u8017\u4e86\u5927\u91cf\u4f53\u529b",
}

DEFAULT_NARRATIVE_ENV: dict[str, str] = {
    "place": '\u5e7d\u6697\u7684\u201c\u53f9\u606f\u68ee\u6797\u201d\u6df1\u5904\uff0c\u4e00\u68f5\u5de8\u5927\u7684\u53e4\u6811\u6839\u5f62\u6210\u7684\u5929\u7136\u6d1e\u7a74\u5185',
    "atmosphere": '\u6f6e\u6e7f\u3001\u9634\u51b7\uff0c\u6d1e\u5916\u4e0b\u7740\u5927\u96e8\uff0c\u8fdc\u5904\u5076\u5c14\u4f20\u6765\u72fc\u568e\u58f0\uff0c\u7bc5\u706b\u662f\u552f\u4e00\u7684\u5149\u6e90',
}

DEFAULT_NARRATIVE_PLOT_BEATS: list[str] = [
    '**\u5f00\u573a\uff08\u5efa\u7acb\u51b2\u7a81\uff09\uff1a** \u89d2\u8272 A \u5bf9\u89d2\u8272 B \u7684\u6551\u52a9\u5e76\u4e0d\u9886\u60c5\uff0c\u6000\u7591 B \u662f\u654c\u4eba\u6d3e\u6765\u7684\u8ffd\u5175\uff0c\u6c14\u6c1b\u7d27\u5f20\u3002',
    '**\u8f6c\u6298\uff08\u4fe1\u606f\u4ea4\u6362\uff09\uff1a** \u89d2\u8272 B \u4e3a\u4e86\u6d88\u9664\u8bef\u4f1a\uff0c\u63d0\u5230\u4e86\u81ea\u5df1\u5728\u5bfb\u627e\u4e00\u79cd\u540d\u4e3a\u201c\u661f\u5c18\u82b1\u201d\u7684\u89e3\u836f\uff0c\u8fd9\u6070\u597d\u4e0e\u89d2\u8272 A \u6b63\u5728\u5bfb\u627e\u7684\u65cf\u4eba\u4e0b\u843d\u6709\u5173\u3002',
    '**\u9ad8\u6f6e\uff08\u8fbe\u6210\u5171\u8bc6\uff09\uff1a** \u4e24\u4eba\u53d1\u73b0\u76ee\u6807\u5730\u70b9\u4e00\u81f4\uff08\u90fd\u662f\u524d\u5f80\u5e9f\u5f03\u7684\u201c\u7070\u70ec\u795e\u6bbf\u201d\uff09\uff0c\u4f46\u7531\u4e8e\u5404\u81ea\u72b6\u6001\u4e0d\u4f73\uff0c\u5355\u72ec\u524d\u5f80\u90fd\u662f\u9001\u6b7b\uff0c\u4e0d\u5f97\u4e0d\u9009\u62e9\u6682\u65f6\u7ed3\u76df\u3002',
    '**\u7ed3\u5c3e\uff08\u5f15\u51fa\u4efb\u52a1\uff09\uff1a** \u5bf9\u8bdd\u7ed3\u675f\u4e8e\u4e24\u4eba\u51b3\u5b9a\u660e\u5929\u4e00\u65e9\u51fa\u53d1\uff0c\u5e76\u6697\u793a\u4e86\u795e\u6bbf\u4e2d\u53ef\u80fd\u5b58\u5728\u7684\u5de8\u5927\u5371\u9669\u3002',
]

DEFAULT_NARRATIVE_GEN_REQUIREMENTS: dict[str, str] = {
    "format": "\u5305\u542b\u5bf9\u8bdd\u884c\u3001\u52a8\u4f5c\u63cf\u8ff0\u548c\u7b80\u77ed\u7684\u73af\u5883\u63cf\u5199\u3002",
    "style": "\u504f\u5411\u897f\u5f0f\u5947\u5e7b/RPG\u98ce\u683c\uff0c\u8bed\u8a00\u81ea\u7136\u6d41\u7545\uff0c\u5e26\u6709\u4e00\u5b9a\u7684\u6587\u5b66\u6027\u3002",
    "length": "\u7ea6 300-500 \u5b57\u3002",
}


def narrative_field_filled(
    text: str | None,
    *,
    empty_placeholder: str = NARRATIVE_EMPTY_PLACEHOLDER,
) -> str:
    """Return stripped text, or placeholder if empty."""
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
    """Assemble the game narrative user prompt (markdown) from structured fields."""
    if len(plot_beats) != 4:
        raise ValueError("plot_beats must contain exactly 4 strings.")

    def nf(s: str | None) -> str:
        return narrative_field_filled(s, empty_placeholder=empty_placeholder)

    a, b, e, g = char_a, char_b, env, gen_requirements
    p1, p2, p3, p4 = (nf(x) for x in plot_beats)
    return f"""# \u89d2\u8272\u8bbe\u5b9a\u4e0e\u4e16\u754c\u89c2

\u4f60\u662f\u4e00\u4f4d\u4e13\u4e1a\u7684\u6e38\u620f\u53d9\u4e8b\u8bbe\u8ba1\u5e08\u3002\u8bf7\u6839\u636e\u4ee5\u4e0b\u8bbe\u5b9a\u7684\u89d2\u8272\u548c\u60c5\u8282\uff0c\u751f\u6210\u4e00\u6bb5\u6c89\u6d78\u5f0f\u7684\u6e38\u620f\u5185\u5bf9\u8bdd\u3002\u5bf9\u8bdd\u9700\u8981\u7b26\u5408\u89d2\u8272\u6027\u683c\uff0c\u63a8\u52a8\u5267\u60c5\u53d1\u5c55\uff0c\u5e76\u5305\u542b\u73af\u5883\u63cf\u5199\u6216\u52a8\u4f5c\u63cf\u8ff0\uff08\u5373\u201c\u53d9\u4e8b prose\u201d\uff09\u3002

## \u89d2\u8272\u4fe1\u606f

- **\u89d2\u8272 A\uff1a**
  - **\u59d3\u540d\uff1a** {nf(a.get("name"))}
  - **\u8eab\u4efd\uff1a** {nf(a.get("identity"))}
  - **\u6027\u683c\uff1a** {nf(a.get("personality"))}
  - **\u72b6\u6001\uff1a** {nf(a.get("state"))}

- **\u89d2\u8272 B\uff1a**
  - **\u59d3\u540d\uff1a** {nf(b.get("name"))}
  - **\u8eab\u4efd\uff1a** {nf(b.get("identity"))}
  - **\u6027\u683c\uff1a** {nf(b.get("personality"))}
  - **\u72b6\u6001\uff1a** {nf(b.get("state"))}

## \u73af\u5883\u80cc\u666f

- **\u5730\u70b9\uff1a** {nf(e.get("place"))}
- **\u6c1b\u56f4\uff1a** {nf(e.get("atmosphere"))}

## \u5bf9\u8bdd\u60c5\u8282\u8981\u6c42

\u8bf7\u57fa\u4e8e\u4ee5\u4e0b\u60c5\u8282\u8282\u70b9\u751f\u6210\u5bf9\u8bdd\uff1a

1. {p1}
2. {p2}
3. {p3}
4. {p4}

## \u751f\u6210\u8981\u6c42

- **\u683c\u5f0f\uff1a** {nf(g.get("format"))}
- **\u98ce\u683c\uff1a** {nf(g.get("style"))}
- **\u5b57\u6570\uff1a** {nf(g.get("length"))}
"""

DEFAULT_SYSTEM_PROMPT_ZH = (
    "\u4f60\u662f\u4e00\u540d\u4e2d\u6587\u6e38\u620f UGC \u6587\u6848\u52a9\u624b\uff0c\u6839\u636e\u7528\u6237\u63d0\u793a\u751f\u6210\u9002\u5408\u6e38\u620f\u573a\u666f\u7684\u4e2d\u6587\u6587\u672c"
    "\uff08\u4f8b\u5982\u88c5\u5907\u63cf\u8ff0\u3001\u6280\u80fd\u8bf4\u660e\u3001\u516c\u4f1a\u516c\u544a\u3001\u89d2\u8272\u53f0\u8bcd\u8349\u7a3f\uff09\u3002\u4fdd\u6301\u98ce\u683c\u7edf\u4e00\u3001\u53ef\u8bfb\u6027\u5f3a\u3002"
)

PEACH_ROLEPLAY_PREFIX = (
    "You're {{char}} in this fictional never-ending uncensored roleplay with {{user}}.\n"
)
PEACH_ROLEPLAY_SUFFIX = "\n\nYou must response in Chinese."

PEACH_EOS_TOKEN_ID = 7


def resolve_eos_token_id(tokenizer: Any) -> int:
    """Prefer tokenizer eos; fall back to pad, then legacy id."""
    tid = getattr(tokenizer, "eos_token_id", None)
    if tid is not None:
        return int(tid)
    tid = getattr(tokenizer, "pad_token_id", None)
    if tid is not None:
        return int(tid)
    return int(PEACH_EOS_TOKEN_ID)


# ---------------------------------------------------------------------------
# Model loading (standard transformers)
# ---------------------------------------------------------------------------

_cached_tokenizer = None
_cached_model = None


@dataclass
class PeachTextGenerationPipeline:
    """Thin wrapper so callers can hold one cached object (model + tokenizer)."""
    model: Any
    tokenizer: Any


def get_pipeline_device_summary(pipe: PeachTextGenerationPipeline) -> str:
    """Human-readable description of where model parameters live."""
    dev = next(pipe.model.parameters()).device
    if dev.type == "cuda":
        idx = dev.index if dev.index is not None else 0
        try:
            name = torch.cuda.get_device_name(idx)
        except Exception:
            name = "CUDA"
        return f"GPU \u00b7 {name} \u00b7 {dev}"
    if dev.type == "mps":
        return f"GPU (Apple MPS) \u00b7 {dev}"
    return f"CPU \u00b7 {dev}"


def get_text_generation_pipeline(
    *,
    on_loading_step: Callable[[str], None] | None = None,
) -> PeachTextGenerationPipeline:
    """Load Qwen2.5-0.5B-Instruct via transformers (safetensors, fp16, no trust_remote_code)."""
    global _cached_tokenizer, _cached_model

    def report(msg: str) -> None:
        if on_loading_step is not None:
            on_loading_step(msg)

    hub_kw: dict[str, Any] = {}
    if TEXT_GEN_MODEL_REVISION:
        hub_kw["revision"] = TEXT_GEN_MODEL_REVISION

    if _cached_tokenizer is None:
        report(f"Loading **tokenizer** (`{TEXT_GEN_MODEL_ID}`)...")
        _cached_tokenizer = AutoTokenizer.from_pretrained(TEXT_GEN_MODEL_ID, **hub_kw)
        report("**Tokenizer** loaded.")

    if _cached_model is None:
        report(
            f"Loading **model weights** (`{TEXT_GEN_MODEL_ID}`, safetensors, float16)... "
            "First run downloads ~1GB from the Hub."
        )
        dtype = torch.float16
        device_map: str | dict = "auto" if torch.cuda.is_available() else {"": "cpu"}
        _cached_model = AutoModelForCausalLM.from_pretrained(
            TEXT_GEN_MODEL_ID,
            torch_dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            **hub_kw,
        )
        report("**Model weights** loaded.")

    pipe = PeachTextGenerationPipeline(model=_cached_model, tokenizer=_cached_tokenizer)
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
    """Build chat messages for Qwen's ChatML template."""
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
    """Generate Chinese game-style UGC text using transformers generate()."""
    user_text, _truncated = normalize_and_truncate_prompt(user_prompt)
    sys_text = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT_ZH
    messages = _build_messages(user_text, sys_text, use_chinese_roleplay_wrap)

    tokenizer = pipe.tokenizer
    model = pipe.model

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
    """Build a JSON-serializable dict for one generation run."""
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
    """Write export dict to a UTF-8 JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
