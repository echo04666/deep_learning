"""
Streamlit app: Chinese game UGC text generation with Peach (HF Hub).

Course layout: constants / loaders / inference helpers live in model_utils.py.
Safety classification pipeline will be added in a later milestone (docs/01).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import streamlit as st

# Streamlit requires set_page_config before any other st.* calls (including @st.cache_resource).
st.set_page_config(page_title="Game UGC — Text Generation", layout="wide")
# #region agent log
try:
    import json as _json
    import time as _time

    with open(
        "/Users/nijialu/Desktop/dl/.cursor/debug-18a7ba.log", "a", encoding="utf-8"
    ) as _df:
        _df.write(
            _json.dumps(
                {
                    "sessionId": "18a7ba",
                    "hypothesisId": "H1",
                    "location": "app.py:after_set_page_config",
                    "message": "set_page_config ok",
                    "data": {},
                    "timestamp": int(_time.time() * 1000),
                }
            )
            + "\n"
        )
except OSError:
    pass
# #endregion

from model_utils import (
    MAX_PROMPT_CHARS,
    DEFAULT_NARRATIVE_CHAR_A,
    DEFAULT_NARRATIVE_CHAR_B,
    DEFAULT_NARRATIVE_ENV,
    DEFAULT_NARRATIVE_GEN_REQUIREMENTS,
    DEFAULT_NARRATIVE_PLOT_BEATS,
    TEXT_GEN_MODEL_ID,
    build_generation_export_dict,
    build_narrative_user_prompt,
    generate_ugc_text,
    get_pipeline_device_summary,
    get_text_generation_pipeline,
    normalize_and_truncate_prompt,
    resolve_eos_token_id,
)


@st.cache_resource(
    show_spinner=(
        "Loading language model — first run downloads several GB from Hugging Face "
        "(often 10–25+ minutes on Community Cloud CPU; keep this tab open, check Logs for tqdm)"
    ),
)
def _load_text_generation_pipeline_cached():
    """One load per Cloud replica; avoids reloading on every Streamlit rerun."""
    # #region agent log
    try:
        import json as _json
        import time as _time

        with open(
            "/Users/nijialu/Desktop/dl/.cursor/debug-18a7ba.log", "a", encoding="utf-8"
        ) as _df:
            _df.write(
                _json.dumps(
                    {
                        "sessionId": "18a7ba",
                        "hypothesisId": "H1",
                        "location": "app.py:_load_text_generation_pipeline_cached",
                        "message": "cache_resource load entered",
                        "data": {},
                        "timestamp": int(_time.time() * 1000),
                    }
                )
                + "\n"
            )
    except OSError:
        pass
    # #endregion
    return get_text_generation_pipeline()


st.title("Game UGC — Text Generation")
st.info(
    "This page runs the **text-generation** step only. "
    "A second Hugging Face pipeline (toxic / safety **classification**) will be added here "
    "in a later milestone (see project docs)."
)
st.warning(
    f"Current Hub model: **{TEXT_GEN_MODEL_ID}**. Large models need a **CUDA GPU** with enough VRAM; "
    "Streamlit Cloud free tier may not load them."
)

if "last_export" not in st.session_state:
    st.session_state.last_export = None

with st.sidebar:
    st.header("Generation settings")
    max_new_tokens = st.slider("max_new_tokens", 64, 1024, 512, 64)
    temperature = st.slider("temperature", 0.1, 1.5, 0.5, 0.05)
    top_p = st.slider("top_p", 0.1, 1.0, 0.7, 0.05)
    do_sample = st.checkbox("do_sample (uncheck for greedy decoding)", value=True)
    use_cn_wrap = st.checkbox(
        "Chinese roleplay wrap on system prompt (model-card README)",
        value=False,
        help="Prepends/appends the official prefix and suffix from the model card to bias Chinese roleplay style.",
    )
    st.subheader("System prompt")
    system_custom = st.text_area(
        "Custom system prompt (optional)",
        value="",
        height=120,
        help="Leave empty to use the default Chinese UGC assistant system prompt in model_utils.",
    )

st.subheader("User prompt")

prompt_source = st.radio(
    "Prompt source",
    ("Free-form text", "Narrative template (same builder as course notebook)"),
    horizontal=True,
    help="Narrative template calls model_utils.build_narrative_user_prompt — same logic as notebooks/04_text_generation_peach_pipeline.ipynb.",
)

free_form_prompt = ""
if prompt_source == "Free-form text":
    free_form_prompt = st.text_area(
        "Prompt for game UGC text (Chinese or bilingual)",
        height=160,
        placeholder="e.g. Write a legendary two-handed sword item description in a wuxia tone.",
    )
else:
    st.caption(
        "Fields are assembled into one markdown user message via `build_narrative_user_prompt` in `model_utils.py`."
    )
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Character A**")
        na = st.text_area(
            "Name",
            value=DEFAULT_NARRATIVE_CHAR_A["name"],
            height=68,
            key="nar_a_name",
        )
        ia = st.text_area(
            "Identity / role",
            value=DEFAULT_NARRATIVE_CHAR_A["identity"],
            height=68,
            key="nar_a_identity",
        )
        pa = st.text_area(
            "Personality",
            value=DEFAULT_NARRATIVE_CHAR_A["personality"],
            height=68,
            key="nar_a_personality",
        )
        sa = st.text_area(
            "Current state",
            value=DEFAULT_NARRATIVE_CHAR_A["state"],
            height=68,
            key="nar_a_state",
        )
    with col_b:
        st.markdown("**Character B**")
        nb = st.text_area(
            "Name",
            value=DEFAULT_NARRATIVE_CHAR_B["name"],
            height=68,
            key="nar_b_name",
        )
        ib = st.text_area(
            "Identity / role",
            value=DEFAULT_NARRATIVE_CHAR_B["identity"],
            height=68,
            key="nar_b_identity",
        )
        pb = st.text_area(
            "Personality",
            value=DEFAULT_NARRATIVE_CHAR_B["personality"],
            height=68,
            key="nar_b_personality",
        )
        sb = st.text_area(
            "Current state",
            value=DEFAULT_NARRATIVE_CHAR_B["state"],
            height=68,
            key="nar_b_state",
        )

    st.markdown("**Setting**")
    ep = st.text_area(
        "Location",
        value=DEFAULT_NARRATIVE_ENV["place"],
        height=80,
        key="nar_env_place",
    )
    ea = st.text_area(
        "Atmosphere",
        value=DEFAULT_NARRATIVE_ENV["atmosphere"],
        height=80,
        key="nar_env_atmosphere",
    )

    st.markdown("**Plot beats (four)**")
    p1 = st.text_area(
        "Beat 1",
        value=DEFAULT_NARRATIVE_PLOT_BEATS[0],
        height=72,
        key="nar_p1",
    )
    p2 = st.text_area(
        "Beat 2",
        value=DEFAULT_NARRATIVE_PLOT_BEATS[1],
        height=72,
        key="nar_p2",
    )
    p3 = st.text_area(
        "Beat 3",
        value=DEFAULT_NARRATIVE_PLOT_BEATS[2],
        height=72,
        key="nar_p3",
    )
    p4 = st.text_area(
        "Beat 4",
        value=DEFAULT_NARRATIVE_PLOT_BEATS[3],
        height=72,
        key="nar_p4",
    )

    st.markdown("**Output requirements**")
    gf = st.text_area(
        "Format",
        value=DEFAULT_NARRATIVE_GEN_REQUIREMENTS["format"],
        height=68,
        key="nar_g_format",
    )
    gs = st.text_area(
        "Style",
        value=DEFAULT_NARRATIVE_GEN_REQUIREMENTS["style"],
        height=68,
        key="nar_g_style",
    )
    gl = st.text_area(
        "Length / word count",
        value=DEFAULT_NARRATIVE_GEN_REQUIREMENTS["length"],
        height=68,
        key="nar_g_length",
    )

def _clear_last_export() -> None:
    st.session_state["last_export"] = None


col_run, col_clear = st.columns(2)
run_clicked = col_run.button("Generate", type="primary")
col_clear.button("Clear last JSON export", on_click=_clear_last_export)
st.caption(
    "First **Generate** pulls **Index-1.9B** (~4GB+) into memory. On **Streamlit Community Cloud** "
    "this can take **many minutes**; the UI may look stuck while **Logs** (Manage app) show download progress. "
    "In **Advanced settings**, pick **Python 3.12** if builds fail on 3.14."
)

if run_clicked:
    st.session_state.last_export = None
    if prompt_source == "Free-form text":
        raw_user_prompt = free_form_prompt
    else:
        try:
            raw_user_prompt = build_narrative_user_prompt(
                char_a={
                    "name": na,
                    "identity": ia,
                    "personality": pa,
                    "state": sa,
                },
                char_b={
                    "name": nb,
                    "identity": ib,
                    "personality": pb,
                    "state": sb,
                },
                env={"place": ep, "atmosphere": ea},
                plot_beats=[p1, p2, p3, p4],
                gen_requirements={"format": gf, "style": gs, "length": gl},
            )
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

    try:
        trimmed, truncated = normalize_and_truncate_prompt(raw_user_prompt)
    except ValueError:
        st.error("Please enter a non-empty prompt.")
        st.stop()

    if truncated:
        st.warning(
            f"Prompt was truncated to {len(trimmed)} characters "
            f"(max {MAX_PROMPT_CHARS}) to protect context length."
        )

    system_prompt = system_custom.strip() or None

    try:
        # Cached load: Streamlit shows a long-running spinner on first miss (Cloud-friendly).
        pipe = _load_text_generation_pipeline_cached()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    st.info(f"**Inference device:** `{get_pipeline_device_summary(pipe)}`")

    gen_params = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": do_sample,
        "repetition_penalty": 1.05,
        "eos_token_id": resolve_eos_token_id(pipe.tokenizer),
        "use_chinese_roleplay_wrap": use_cn_wrap,
    }

    try:
        with st.spinner("Generating text..."):
            generated_text, elapsed_sec, messages = generate_ugc_text(
                pipe,
                trimmed,
                system_prompt=system_prompt,
                use_chinese_roleplay_wrap=use_cn_wrap,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )
    except Exception as exc:
        st.error(f"Generation failed: {exc}")
        st.stop()

    st.success(f"Done in {elapsed_sec:.2f} s")
    st.subheader("Generated text")
    st.write(generated_text)

    export = build_generation_export_dict(
        generated_text=generated_text,
        messages=messages,
        generation_params=gen_params,
        elapsed_sec=elapsed_sec,
        model_id=TEXT_GEN_MODEL_ID,
    )
    st.session_state.last_export = export

if st.session_state.last_export is not None:
    st.subheader("Export JSON")
    payload = st.session_state.last_export
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    file_name = f"peach_generation_{ts}.json"
    st.download_button(
        label="Download JSON",
        data=json.dumps(payload, ensure_ascii=False, indent=2),
        file_name=file_name,
        mime="application/json",
    )
    with st.expander("Preview JSON"):
        st.json(payload)
