"""
Streamlit app: Chinese game UGC text generation with Qwen2.5-0.5B-Instruct (transformers).

Course layout: constants / loaders / inference helpers live in model_utils.py.
Safety classification pipeline will be added in a later milestone (docs/01).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import streamlit as st

st.set_page_config(page_title="Game UGC \u2014 Text Generation", layout="wide")

from model_utils import (
    DEFAULT_SYSTEM_PROMPT_ZH,
    MAX_PROMPT_CHARS,
    DEFAULT_NARRATIVE_CHAR_A,
    DEFAULT_NARRATIVE_CHAR_B,
    DEFAULT_NARRATIVE_ENV,
    DEFAULT_NARRATIVE_GEN_REQUIREMENTS,
    DEFAULT_NARRATIVE_PLOT_BEATS,
    DEFAULT_NARRATIVE_SYSTEM_PROMPT_ZH,
    TEXT_GEN_MODEL_ID,
    build_generation_export_dict,
    build_narrative_user_prompt,
    generate_ugc_text,
    get_text_generation_pipeline,
    normalize_and_truncate_prompt,
    resolve_eos_token_id,
)


@st.cache_resource(
    show_spinner=(
        "Loading Qwen2.5-0.5B-Instruct \u2014 first run downloads ~1GB from Hugging Face "
        "(Community Cloud CPU: may take a few minutes; keep this tab open, check Logs)"
    ),
)
def _load_text_generation_pipeline_cached():
    """One load per Cloud replica; avoids reloading on every Streamlit rerun."""
    return get_text_generation_pipeline()


st.title("Game UGC \u2014 Text Generation")

if "last_export" not in st.session_state:
    st.session_state.last_export = None

with st.sidebar:
    st.header("Generation settings")
    do_sample = st.checkbox(
        "Creative mode",
        value=True,
        help="Turn on for more varied output. Turn off for more stable and repeatable output.",
    )
    with st.expander("What changes when Creative mode is on/off?"):
        st.markdown(
            "- **Checked (Creative mode ON):** uses probabilistic sampling. Results vary across runs and can be more imaginative.\n"
            "- **Unchecked (Creative mode OFF):** uses deterministic decoding. Results are more stable and repeatable.\n"
            "- When OFF, **temperature** and **top-p** are disabled because they only affect sampling."
        )
    max_new_tokens = st.slider(
        "Maximum response length (tokens)",
        64,
        1024,
        512,
        64,
        help="Upper limit for generated output length. Higher values can produce longer text but take more time.",
    )
    temperature = st.slider(
        "Creativity level (temperature)",
        0.1,
        1.5,
        0.5,
        0.05,
        disabled=not do_sample,
        help="Lower values are more focused and deterministic; higher values are more diverse and creative.",
    )
    top_p = st.slider(
        "Word-choice diversity (top-p)",
        0.1,
        1.0,
        0.7,
        0.05,
        disabled=not do_sample,
        help="Controls how broad the candidate token pool is. Lower values are safer; higher values allow more variety.",
    )
    use_cn_wrap = st.checkbox(
        "Dialogue mode",
        value=False,
        help="Wraps the system prompt with roleplay instructions so outputs are more dialogue-like and in-character.",
    )
    st.subheader("System prompt")
    system_custom = st.text_area(
        "Custom system prompt (optional)",
        value="",
        height=120,
        help=(
            "Use this when you want strict style/format control (tone, persona, output constraints). "
            "Leave empty to use defaults.\n\n"
            f"Default for Free-form:\n{DEFAULT_SYSTEM_PROMPT_ZH}\n\n"
            f"Default for Narrative template:\n{DEFAULT_NARRATIVE_SYSTEM_PROMPT_ZH}"
        ),
    )

st.subheader("User prompt")

prompt_source = st.radio(
    "Prompt source",
    ("Free-form mode", "Narrative template"),
    horizontal=True,
    help="Narrative template calls model_utils.build_narrative_user_prompt.",
)

free_form_prompt = ""
if prompt_source == "Free-form mode":
    free_form_prompt = st.text_area(
        "Prompt for game UGC text (Chinese)",
        height=160,
        placeholder="e.g. Write a legendary two-handed sword item description in a wuxia tone.",
    )
else:
    st.caption(
        "Fill in the fields below and the text will be assembled for dialogue generation"
    )
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Character A**")
        na = st.text_area("Name", value=DEFAULT_NARRATIVE_CHAR_A["name"], height=68, key="nar_a_name")
        ia = st.text_area("Identity / role", value=DEFAULT_NARRATIVE_CHAR_A["identity"], height=68, key="nar_a_identity")
        pa = st.text_area("Personality", value=DEFAULT_NARRATIVE_CHAR_A["personality"], height=68, key="nar_a_personality")
        sa = st.text_area("Current state", value=DEFAULT_NARRATIVE_CHAR_A["state"], height=68, key="nar_a_state")
    with col_b:
        st.markdown("**Character B**")
        nb = st.text_area("Name", value=DEFAULT_NARRATIVE_CHAR_B["name"], height=68, key="nar_b_name")
        ib = st.text_area("Identity / role", value=DEFAULT_NARRATIVE_CHAR_B["identity"], height=68, key="nar_b_identity")
        pb = st.text_area("Personality", value=DEFAULT_NARRATIVE_CHAR_B["personality"], height=68, key="nar_b_personality")
        sb = st.text_area("Current state", value=DEFAULT_NARRATIVE_CHAR_B["state"], height=68, key="nar_b_state")

    st.markdown("**Setting**")
    ep = st.text_area("Location", value=DEFAULT_NARRATIVE_ENV["place"], height=80, key="nar_env_place")
    ea = st.text_area("Atmosphere", value=DEFAULT_NARRATIVE_ENV["atmosphere"], height=80, key="nar_env_atmosphere")

    st.markdown("**Plot beats (four)**")
    p1 = st.text_area("Beat 1", value=DEFAULT_NARRATIVE_PLOT_BEATS[0], height=72, key="nar_p1")
    p2 = st.text_area("Beat 2", value=DEFAULT_NARRATIVE_PLOT_BEATS[1], height=72, key="nar_p2")
    p3 = st.text_area("Beat 3", value=DEFAULT_NARRATIVE_PLOT_BEATS[2], height=72, key="nar_p3")
    p4 = st.text_area("Beat 4", value=DEFAULT_NARRATIVE_PLOT_BEATS[3], height=72, key="nar_p4")

    st.markdown("**Output requirements**")
    gf = st.text_area("Format", value=DEFAULT_NARRATIVE_GEN_REQUIREMENTS["format"], height=68, key="nar_g_format")
    gs = st.text_area("Style", value=DEFAULT_NARRATIVE_GEN_REQUIREMENTS["style"], height=68, key="nar_g_style")
    gl = st.text_area("Length / word count", value=DEFAULT_NARRATIVE_GEN_REQUIREMENTS["length"], height=68, key="nar_g_length")

def _clear_last_export() -> None:
    st.session_state["last_export"] = None


col_run, col_clear = st.columns(2)
run_clicked = col_run.button("Generate", type="primary")
col_clear.button("Clear last JSON export", on_click=_clear_last_export)
st.caption(
    "The first load will require downloading the model, please be patient ~"
)

if run_clicked:
    st.session_state.last_export = None
    if prompt_source == "Free-form mode":
        raw_user_prompt = free_form_prompt
    else:
        try:
            raw_user_prompt = build_narrative_user_prompt(
                char_a={"name": na, "identity": ia, "personality": pa, "state": sa},
                char_b={"name": nb, "identity": ib, "personality": pb, "state": sb},
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
        prompt_length_status = (
            f"Truncated to {len(trimmed)} characters "
            f"(limit: {MAX_PROMPT_CHARS})."
        )
        st.warning(f"Prompt length status: {prompt_length_status}")
    else:
        prompt_length_status = (
            f"Within limit ({len(trimmed)}/{MAX_PROMPT_CHARS} characters)."
        )
        st.info(f"Prompt length status: {prompt_length_status}")

    if system_custom.strip():
        system_prompt = system_custom.strip()
    elif prompt_source == "Narrative template (same builder as course notebook)":
        system_prompt = DEFAULT_NARRATIVE_SYSTEM_PROMPT_ZH
    else:
        system_prompt = None

    narrative_mode = prompt_source == "Narrative template (same builder as course notebook)"
    repetition_penalty = 1.15 if narrative_mode else 1.05
    no_repeat_ngram = 3 if narrative_mode else None

    loading_status = st.empty()
    loading_status.info("Start loading model...")

    try:
        pipe = _load_text_generation_pipeline_cached()
    except RuntimeError as exc:
        loading_status.empty()
        st.error(str(exc))
        st.stop()
    loading_status.success("Model loaded successfully.")

    gen_params = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": do_sample,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram,
        "eos_token_id": resolve_eos_token_id(pipe.tokenizer),
        "use_chinese_roleplay_wrap": use_cn_wrap,
        "prompt_length_status": prompt_length_status,
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
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram,
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
    file_name = f"qwen_generation_{ts}.json"
    st.download_button(
        label="Download JSON",
        data=json.dumps(payload, ensure_ascii=False, indent=2),
        file_name=file_name,
        mime="application/json",
    )
    with st.expander("Preview JSON"):
        st.json(payload)
