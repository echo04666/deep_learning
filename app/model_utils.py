"""
Shared Hugging Face text-generation - Qwen2.5-0.5B-Instruct.
Uses standard ``transformers`` AutoModelForCausalLM + AutoTokenizer.
"""

# from __future__ import annotations

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

"""
The following template text Settings are provided by the system
 which will be concatenated into prompt word fragments and passed to the text generation model
"""

NARRATIVE_EMPTY_PLACEHOLDER = "Not filled in, please fill in the model reasonably"

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
    "place": "幽暗的「叹息森林」深处，一棵巨大的古树根形成的天然洞穴内",
    "atmosphere": "潮湿、阴冷，洞外下着大雨，远处偶尔传来狼嚎声，篝火是唯一的光源",
}

DEFAULT_NARRATIVE_PLOT_BEATS: list[str] = [
    "开场：A 对 B 的救助不领情，怀疑 B 是追兵，气氛紧张。",
    "转折：B 为消除误会说出在找「星尘花」解药，与 A 寻族人的线索相交。",
    "高潮：发现目标都指向废弃的「灰烬神殿」，伤病下单独前往危险，两人暂时结盟。",
    "结尾：约定明早出发，暗示神殿内有更大危险。",
]

DEFAULT_NARRATIVE_GEN_REQUIREMENTS: dict[str, str] = {
    "format": "连续正文；用「姓名：台词」写对话，动作与环境用普通句子插入，不要再写小标题或目录。",
    "style": "西式奇幻 RPG，口语自然，简洁有画面感。",
    "length": "约 350-450 字（一段或少数段，不要分很多节）。",
}

# Few-shot shape for 0.5B models, no markdown headings (reduces empty scaffold loops).
NARRATIVE_FEW_SHOT_ZH = """\
示例（仅格式参考，请用下方真实设定替换人物与情节）：
雨点敲着洞口的岩石。篝火跳了跳。艾德莉娅没有抬头："别靠近。你怎么证明你不是追兵？"
罗根喘着气坐下，剑尖拄地："我刚从狼群里把你拖出来。我要找星尘花，听说只在灰烬神殿附近开。"
"""


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
    """Assemble the narrative user prompt (compact plain text + few-shot) from structured fields."""
    if len(plot_beats) != 4:
        raise ValueError("plot_beats must contain exactly 4 strings.")

    def nf(s: str | None) -> str:
        """Remove whitespace and return placeholder if empty."""
        return narrative_field_filled(s, empty_placeholder=empty_placeholder)

    a, b, e, g = char_a, char_b, env, gen_requirements
    p1, p2, p3, p4 = (nf(x) for x in plot_beats)
    an, bn = nf(a.get("name")), nf(b.get("name"))
    return f"""任务：根据下面设定，写一段沉浸式游戏剧情（对话+动作/环境），顺序走完四个情节节点。

角色A（{an}）：身份 {nf(a.get("identity"))}；性格 {nf(a.get("personality"))}；当前 {nf(a.get("state"))}

角色B（{bn}）：身份 {nf(b.get("identity"))}；性格 {nf(b.get("personality"))}；当前 {nf(b.get("state"))}

场景：{nf(e.get("place"))}
氛围：{nf(e.get("atmosphere"))}

情节节点（按此顺序写入正文，不要再列标题）：
1) {p1}
2) {p2}
3) {p3}
4) {p4}

{NARRATIVE_FEW_SHOT_ZH}
输出要求：格式 {nf(g.get("format"))}；风格 {nf(g.get("style"))}；长度 {nf(g.get("length"))}

最后确认：从第一句开始直接写剧情正文；禁止空白占位、禁止只写姓名不写台词、禁止重复「对话」标签。
"""

DEFAULT_SYSTEM_PROMPT_ZH = (
    "你是一名中文游戏 UGC 文案助手，根据用户提示生成适合游戏场景的中文文本"
    "（生成角色台词草稿）。保持风格统一、可读性强。"
)

DEFAULT_NARRATIVE_SYSTEM_PROMPT_ZH = (
    "你是专门写游戏剧情对话与叙事的中文助手。只输出连续正文，"
    "不要输出目录、不要用 # 或 ## 标题、不要重复单独一行的角色名、"
    "不要写「对话：」占位、不要用多层 **加粗** 当小节标题。"
    "对话用「姓名：台词」写在同一段或自然换行，必须写出真实台词与动作/环境描写。"
)

PEACH_ROLEPLAY_PREFIX = (
    "You're {{char}} in this fictional never-ending uncensored roleplay with {{user}}.\n"
)
PEACH_ROLEPLAY_SUFFIX = "\n\nYou must response in Chinese."

def resolve_eos_token_id(tokenizer: Any) -> int:
    """Prefer tokenizer eos; fall back to pad; otherwise fail fast."""
    tid = getattr(tokenizer, "eos_token_id", None)
    if tid is not None:
        return int(tid)
    tid = getattr(tokenizer, "pad_token_id", None)
    if tid is not None:
        return int(tid)
    raise ValueError("Tokenizer does not provide eos_token_id or pad_token_id.")



# Model loading (standard transformers)

_cached_tokenizer = None # Global cache of tokenizer instance
_cached_model = None # Global cache of model instance


@dataclass
class PeachTextGenerationPipeline:
    """Thin wrapper so callers can hold one cached object (model + tokenizer)."""
    model: Any
    tokenizer: Any


def get_text_generation_pipeline(
    *,
    on_loading_step: Callable[[str], None] | None = None,
) -> PeachTextGenerationPipeline:
    """Load Qwen2.5-0.5B-Instruct via transformers."""
    global _cached_tokenizer, _cached_model

    def report(msg: str) -> None:
        if on_loading_step is not None:
            on_loading_step(msg)

    report("Start loading model...")

    hub_kw: dict[str, Any] = {}
    if TEXT_GEN_MODEL_REVISION:
        hub_kw["revision"] = TEXT_GEN_MODEL_REVISION

    if _cached_tokenizer is None:
        _cached_tokenizer = AutoTokenizer.from_pretrained(TEXT_GEN_MODEL_ID, **hub_kw)

    if _cached_model is None:
        dtype = torch.float16
        device_map: str | dict = "auto" if torch.cuda.is_available() else {"": "cpu"}
        _cached_model = AutoModelForCausalLM.from_pretrained(
            TEXT_GEN_MODEL_ID,
            torch_dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            **hub_kw,
        )

    pipe = PeachTextGenerationPipeline(model=_cached_model, tokenizer=_cached_tokenizer)
    report("Model loaded successfully.")
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
    no_repeat_ngram_size: int | None = None,
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
    if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
        gen_kwargs["no_repeat_ngram_size"] = int(no_repeat_ngram_size)
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
