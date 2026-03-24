"""
Streamlit app: Chinese game UGC text generation (Qwen) + safety check (fine-tuned BERT).
Step 1: generate text. 
Step 2: per-sentence safety review (Tencent wordlist from GitHub + HF classifier); publish only when all clear.
History: published texts saved under app/data/publish_history.json (local / Cloud ephemeral).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from model_utils import (
    DEFAULT_SYSTEM_PROMPT_ZH,
    MAX_PROMPT_CHARS,
    DEFAULT_NARRATIVE_CHAR_A,
    DEFAULT_NARRATIVE_CHAR_B,
    DEFAULT_NARRATIVE_ENV,
    DEFAULT_NARRATIVE_GEN_REQUIREMENTS,
    DEFAULT_NARRATIVE_PLOT_BEATS,
    DEFAULT_NARRATIVE_SYSTEM_PROMPT_ZH,
    build_narrative_user_prompt,
    generate_ugc_text,
    get_text_generation_pipeline,
    normalize_and_truncate_prompt,
)
from toxic_classification_pipeline import (
    MAX_MANUAL_SENTENCE_CHARS,
    TOXIC_CLF_MODEL_ID,
    classify_one_sentence_with_wordlist,
    get_text_classification_pipeline,
    is_predicted_toxic,
    new_sentence_item,
    split_into_sentences,
    validate_manual_sentence,
)

# path settings
APP_DIR = Path(__file__).resolve().parent
HISTORY_JSON_PATH = APP_DIR / "data" / "publish_history.json"


def _toxic_sentence_hint_captions(it: dict) -> list[str]:
    """Zh captions under a failed sentence: wordlist → possible terms; model → generic hint only."""
    if not it.get("checked") or not it.get("is_toxic"):
        return []

    model_toxic = it.get("model_is_toxic")
    if model_toxic is None and it.get("label") not in (None, ""):
        model_toxic = is_predicted_toxic(str(it["label"]))
    model_toxic = bool(model_toxic)

    hits = it.get("dict_hits") if isinstance(it.get("dict_hits"), list) else []
    sens = it.get("is_sensitive_hit")
    if sens is None:
        sens = bool(hits)
    sens = bool(sens)

    lines: list[str] = []
    if sens and hits:
        lines.append("可能的敏感词：" + "、".join(hits))
    elif sens:
        lines.append("可能包含敏感词。")

    if model_toxic:
        if sens and hits:
            lines.append("可能包含敏感词。")
        elif not sens:
            lines.append("可能包含敏感词。")

    if not lines:
        lines.append("未通过安全检测，请修改后重新检测。")
    return lines


def _load_publish_history() -> list[dict]:
    """Load past publications from disk."""
    if not HISTORY_JSON_PATH.exists():
        return []
    try:
        with open(HISTORY_JSON_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return _normalize_history_list(data)
        return []
    except (json.JSONDecodeError, OSError):
        return []


def _save_publish_history(entries: list[dict]) -> None:
    """Persist publication history (UTF-8 JSON)."""
    HISTORY_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def build_compliance_payload(
    *,
    full_text: str,
    lines: list[str],
    published_at: str,
    entry_id: str,
) -> dict:
    """Compliance-only export: no scores or raw classifier outputs."""
    return {
        "schema_version": "1.0",
        "status": "all_sentences_cleared",
        "entry_id": entry_id,
        "published_at": published_at,
        "classifier_model_id": TOXIC_CLF_MODEL_ID,
        "full_text": full_text,
        "sentences": [{"index": i + 1, "text": t} for i, t in enumerate(lines)],
    }


def _coerce_history_entry(raw: dict) -> dict:
    """Normalize disk/session entries to include a ``compliance`` block."""
    if raw.get("compliance") and isinstance(raw["compliance"], dict):
        out = dict(raw)
        if not out.get("id"):
            out["id"] = str(uuid.uuid4())
        return out
    ft = raw.get("full_text", "") or ""
    lines = list(raw.get("sentences") or [])
    if not lines and ft.strip():
        lines = [ln for ln in ft.split("\n") if ln.strip()]
    pid = raw.get("id") or str(uuid.uuid4())
    at = raw.get("published_at") or ""
    comp = build_compliance_payload(
        full_text=ft,
        lines=lines,
        published_at=at,
        entry_id=pid,
    )
    return {"id": pid, "published_at": at, "compliance": comp}


def _normalize_history_list(entries: list[dict]) -> list[dict]:
    return [_coerce_history_entry(e) for e in entries if isinstance(e, dict)]


def _render_readonly_history_entry(
    entry: dict,
    *,
    idx_label: str,
    widget_key_prefix: str = "",
) -> None:
    """Show one saved record: read-only text + compliance JSON download."""
    comp = entry.get("compliance") or {}
    full_text = comp.get("full_text", "") or entry.get("full_text", "")
    st.markdown(f"**{idx_label}** · `{entry.get('published_at', '')}`")
    st.text(full_text)
    sents = comp.get("sentences") or []
    if sents:
        st.caption("Sentences")
        for row in sents:
            if isinstance(row, dict):
                st.markdown(f"{row.get('index', '')}. {row.get('text', '')}")
            else:
                st.markdown(f"- {row}")
    payload = json.dumps(comp, ensure_ascii=False, indent=2)
    safe_ts = "".join(c if c.isalnum() else "_" for c in entry.get("published_at", "export"))[:40]
    eid = entry.get("id") or idx_label
    st.download_button(
        label="Download compliance JSON",
        data=payload,
        file_name=f"compliance_{safe_ts}_{str(eid)[:8]}.json",
        mime="application/json",
        key=f"{widget_key_prefix}dl_hist_{eid}",
    )


def _init_session_state() -> None:
    """One-time session defaults."""
    if "wizard_step" not in st.session_state:
        st.session_state.wizard_step = 1
    if "generated_text" not in st.session_state:
        st.session_state.generated_text = ""
    if "safety_items" not in st.session_state:
        st.session_state.safety_items = []
    if "publish_history" not in st.session_state:
        st.session_state.publish_history = _load_publish_history()
    if "prompt_source_mode" not in st.session_state:
        st.session_state.prompt_source_mode = "Narrative template"


@st.cache_resource(
    show_spinner=(
        "Loading Qwen2.5-0.5B-Instruct from Hugging Face "
        "(Community Cloud CPU: may take a few minutes; keep this tab open, check Logs)"
    ),
)
def _load_text_generation_pipeline_cached():
    """One load per Cloud replica; avoids reloading on every Streamlit rerun."""
    return get_text_generation_pipeline()


@st.cache_resource(
    show_spinner=(
        "Loading toxic classification model — first run may download from Hugging Face "
    ),
)
def _load_text_classification_pipeline_cached():
    """Lazy load pipeline 2"""
    return get_text_classification_pipeline()


def _sync_safety_texts_from_widgets() -> None:
    """Copy current text_area values into safety_items (source of truth for logic).

    Reassigns the whole list so Streamlit persists nested updates in st.session_state.
    """
    new_items = []
    for it in st.session_state.safety_items:
        row = dict(it)
        key = f"area_{row['id']}"
        if key in st.session_state:
            row["text"] = str(st.session_state[key]).strip()
        new_items.append(row)
    st.session_state.safety_items = new_items


def _all_sentences_safe_and_checked() -> bool:
    """True only when every line was classified and none is toxic."""
    items = st.session_state.safety_items
    if not items:
        return False
    for it in items:
        if not it.get("text", "").strip():
            return False
        if not it.get("checked"):
            return False
        if it.get("is_toxic"):
            return False
    return True


def _run_classify_on_item(item_id: str) -> None:
    """Classify one row by id after syncing widget text.

    Replaces that row with a new dict and reassigns the list (Streamlit session_state).
    """
    _sync_safety_texts_from_widgets()
    pipe = _load_text_classification_pipeline_cached()
    items = list(st.session_state.safety_items)
    for i, it in enumerate(items):
        if it["id"] != item_id:
            continue
        text = it["text"].strip()
        if not text:
            items[i] = {
                **it,
                "label": "",
                "score": 0.0,
                "is_toxic": False,
                "dict_hits": [],
                "is_sensitive_hit": False,
                "model_is_toxic": False,
                "checked": True,
            }
            st.session_state.safety_items = items
            return
        try:
            out = classify_one_sentence_with_wordlist(pipe, text)
        except Exception as exc:
            st.session_state["_last_cls_error"] = str(exc)
            return
        items[i] = {
            **it,
            "label": out["label"],
            "score": out["score"],
            "is_toxic": out["is_toxic"],
            "dict_hits": out.get("dict_hits", []),
            "is_sensitive_hit": out.get("is_sensitive_hit", False),
            "model_is_toxic": out.get("model_is_toxic", False),
            "checked": True,
        }
        st.session_state.safety_items = items
        return


def _run_classify_all() -> None:
    """Classify every sentence (used by full check button)."""
    _sync_safety_texts_from_widgets()
    pipe = _load_text_classification_pipeline_cached()
    err = None
    items = list(st.session_state.safety_items)
    for i, it in enumerate(items):
        text = it["text"].strip()
        if not text:
            items[i] = {
                **it,
                "label": "",
                "score": 0.0,
                "is_toxic": False,
                "dict_hits": [],
                "is_sensitive_hit": False,
                "model_is_toxic": False,
                "checked": True,
            }
            continue
        try:
            out = classify_one_sentence_with_wordlist(pipe, text)
        except Exception as exc:
            err = str(exc)
            break
        items[i] = {
            **it,
            "label": out["label"],
            "score": out["score"],
            "is_toxic": out["is_toxic"],
            "dict_hits": out.get("dict_hits", []),
            "is_sensitive_hit": out.get("is_sensitive_hit", False),
            "model_is_toxic": out.get("model_is_toxic", False),
            "checked": True,
        }
    st.session_state.safety_items = items
    if err:
        st.session_state["_last_cls_error"] = err


def _remove_sentence_item(item_id: str) -> None:
    """Delete one row and drop its widget key to avoid Streamlit key reuse issues."""
    key = f"area_{item_id}"
    if key in st.session_state:
        del st.session_state[key]
    st.session_state.safety_items = [
        x for x in st.session_state.safety_items if x["id"] != item_id
    ]


def _queue_recheck(item_id: str) -> None:
    """Defer single-sentence classify to the main script so ``st.spinner`` can show."""
    st.session_state["_pending_recheck_id"] = item_id


def main() -> None:
    """Build and run the Streamlit UI (course entry point)."""
    st.set_page_config(
        page_title="Game UGC — Generate & Safety",
        layout="wide",
    )
    _init_session_state()

    # UI
    st.title("Game UGC — Text generation & safety check")
    
    tab_workflow, tab_history = st.tabs(["Generate & safety", "Publication history"])
    
    with tab_workflow:
        step = st.session_state.wizard_step
        if step == 1:
            st.info("Step 1 / 2: Generate UGC text, then go to **Safety check**.")
        else:
            st.info(
                "Step 2 / 2: Review each sentence. **Publish** is enabled only when every line has been checked "
                "and passed the automated safety review."
            )
    
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
    
        st.subheader("Step 1 — User prompt")
    
        prompt_source = st.radio(
            "Prompt source",
            ("Narrative template", "Free-form mode"),
            horizontal=True,
            help="Narrative template calls model_utils.build_narrative_user_prompt.",
            key="prompt_source_mode",
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
    
        col_run, _col_placeholder = st.columns(2)
        run_clicked = col_run.button("Generate", type="primary")
        st.caption("The first load will require downloading the model, please be patient ~")
    
        if run_clicked:
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
            elif prompt_source == "Narrative template":
                system_prompt = DEFAULT_NARRATIVE_SYSTEM_PROMPT_ZH
            else:
                system_prompt = None
    
            narrative_mode = prompt_source == "Narrative template"
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
    
            try:
                with st.spinner("Generating text..."):
                    generated_text, elapsed_sec, _messages = generate_ugc_text(
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
    
            st.session_state.generated_text = generated_text
            st.session_state.wizard_step = 1
    
            st.success(f"Done in {elapsed_sec:.2f} s")
            st.subheader("Generated text")
            st.write(generated_text)
    
        # Show last generated text if we did not just run (rerun / step 2 back)
        elif st.session_state.generated_text and st.session_state.generated_text.strip():
            st.subheader("Current generated text")
            st.write(st.session_state.generated_text)
    
        # Place AFTER generation updates session_state so disabled= is correct on the same run as Generate.
        has_generated = bool((st.session_state.get("generated_text") or "").strip())
        go_safety = st.button(
            "Safety Check (Step 2)",
            type="secondary",
            key="btn_go_safety_step2",
            help="Split the last generated text into sentences and open the safety review step.",
            disabled=not has_generated,
        )
        if go_safety:
            body = st.session_state.generated_text.strip()
            if not body:
                st.error("No generated text yet. Please run **Generate** first.")
            else:
                parts = split_into_sentences(body)
                st.session_state.safety_items = [new_sentence_item(s) for s in parts]
                st.session_state.wizard_step = 2
                st.session_state["_last_cls_error"] = None
                st.rerun()
    
        # --- Step 2: safety ---
        if st.session_state.wizard_step == 2:
            st.divider()
            st.subheader("Step 2 — Safety check (per sentence)")
    
            err_key = "_last_cls_error"
            if err_key in st.session_state and st.session_state[err_key]:
                st.error(f"Classification error: {st.session_state[err_key]}")
                st.session_state[err_key] = None
    
            _pending_recheck = st.session_state.get("_pending_recheck_id")
    
            c_back, c_full, c_pub = st.columns([1, 1, 2])
            if c_back.button("Back to step 1"):
                st.session_state.wizard_step = 1
                st.session_state.safety_items = []
                st.rerun()
            if c_full.button("Run full check on all sentences", type="primary"):
                with st.spinner("Checking all sentences…"):
                    _run_classify_all()
                st.rerun()
    
            publish_ok = _all_sentences_safe_and_checked()
            if c_pub.button(
                "Publish (save to history)",
                type="primary",
                disabled=not publish_ok,
                help="Enabled only when every sentence has been checked and passed automated safety review.",
            ):
                _sync_safety_texts_from_widgets()
                lines = [it["text"].strip() for it in st.session_state.safety_items if it["text"].strip()]
                full_text = "\n".join(lines)
                entry_id = str(uuid.uuid4())
                published_at = datetime.now(timezone.utc).isoformat()
                compliance = build_compliance_payload(
                    full_text=full_text,
                    lines=lines,
                    published_at=published_at,
                    entry_id=entry_id,
                )
                entry = {
                    "id": entry_id,
                    "published_at": published_at,
                    "compliance": compliance,
                }
                # New list assignment so Streamlit persists session_state (avoid .append() on nested list).
                updated_hist = list(st.session_state.publish_history) + [entry]
                st.session_state.publish_history = updated_hist
                try:
                    _save_publish_history(updated_hist)
                except OSError as exc:
                    st.warning(f"Saved in this session only (could not write file): {exc}")
                st.success("Published and saved to history.")
                st.session_state.wizard_step = 1
                st.session_state.safety_items = []
                st.rerun()
    
            if not publish_ok and st.session_state.safety_items:
                st.warning(
                    "Publishing is locked until **every** sentence has been checked and passed the automated "
                    "safety review (use **Run full check** or per-line **Recheck**)."
                )
    
            # New sentence (manual, max 100 chars)
            with st.expander("Add a new sentence (max 100 characters)", expanded=False):
                new_line = st.text_input(
                    "New sentence",
                    max_chars=MAX_MANUAL_SENTENCE_CHARS,
                    key="new_sentence_draft_input",
                )
                if st.button("Add sentence"):
                    ok_text, msg = validate_manual_sentence(new_line)
                    if msg:
                        st.error(msg)
                    else:
                        st.session_state.safety_items = st.session_state.safety_items + [
                            new_sentence_item(ok_text)
                        ]
                        # Widget-bound keys cannot be assigned directly; delete so next run recreates empty.
                        if "new_sentence_draft_input" in st.session_state:
                            del st.session_state["new_sentence_draft_input"]
                        st.session_state["_add_sentence_success_msg"] = "Successful"
                        st.rerun()
    
            _add_msg = st.session_state.pop("_add_sentence_success_msg", None)
            if _add_msg:
                st.success(_add_msg)
    
            for idx, it in enumerate(st.session_state.safety_items):
                st.markdown(f"**Sentence {idx + 1}**")
                col_t, col_meta = st.columns([3, 1])
                with col_t:
                    st.text_area(
                        "Text",
                        value=it["text"],
                        height=100,
                        key=f"area_{it['id']}",
                        label_visibility="collapsed",
                    )
                with col_meta:
                    is_rechecking_this_row = _pending_recheck == it["id"]
                    if it.get("checked"):
                        if it.get("is_toxic"):
                            st.error("Sensitive content, please modify")
                            for _hint in _toxic_sentence_hint_captions(it):
                                st.caption(_hint)
                        else:
                            st.success("OK")
                    else:
                        st.info("Not checked yet")
                    if is_rechecking_this_row:
                        st.caption("Rechecking...")

                    b1, b2 = st.columns(2)
                    # Use on_click so the handler runs reliably; reassign list so session_state updates stick.
                    b1.button(
                        "Rechecking..." if is_rechecking_this_row else "Recheck",
                        key=f"recheck_{it['id']}",
                        on_click=_queue_recheck,
                        args=(it["id"],),
                        disabled=is_rechecking_this_row,
                    )
                    b2.button(
                        "Delete",
                        key=f"del_{it['id']}",
                        on_click=_remove_sentence_item,
                        args=(it["id"],),
                        disabled=is_rechecking_this_row,
                    )
    
            # Run deferred single-row recheck after rendering once so user can see status near the sentence.
            if _pending_recheck:
                _pending_idx = None
                for _i, _it in enumerate(st.session_state.safety_items, start=1):
                    if _it["id"] == _pending_recheck:
                        _pending_idx = _i
                        break
                _label = f"Sentence {_pending_idx}" if _pending_idx is not None else "selected sentence"
                with st.spinner(f"Rechecking {_label} with the classifier..."):
                    _run_classify_on_item(_pending_recheck)
                st.session_state.pop("_pending_recheck_id", None)
                st.rerun()
    
            if not st.session_state.safety_items:
                st.warning("No sentences. Go back to step 1 or add a sentence manually.")
    
        # --- Compliant history on main workflow tab (read-only; fixes visibility after publish)
        _ph = st.session_state.publish_history
        if _ph:
            st.divider()
            st.subheader("Compliant saved records")
            st.caption(
                "Read-only (not editable). Expand a row to view full text and download its compliance JSON."
            )
            for i, entry in enumerate(reversed(_ph)):
                when = entry.get("published_at", "")
                n = len(_ph) - i
                with st.expander(f"{n}. {when}"):
                    _render_readonly_history_entry(
                        entry,
                        idx_label=f"Record #{n}",
                        widget_key_prefix="wf_tab_",
                    )
            _bundle_wf = json.dumps(
                {"records": [e.get("compliance", {}) for e in _ph]},
                ensure_ascii=False,
                indent=2,
            )
            st.download_button(
                label="Download all compliance JSON (bundle)",
                data=_bundle_wf,
                file_name="compliance_history_bundle.json",
                mime="application/json",
                key="wf_tab_dl_compliance_bundle",
            )
    
    with tab_history:
        st.subheader("Publication history")
        hist = st.session_state.publish_history
        if not hist:
            st.info("No publications yet. Complete step 2 and click **Publish (save to history)**.")
        else:
            for i, entry in enumerate(reversed(hist)):
                when = entry.get("published_at", "")
                n = len(hist) - i
                with st.expander(f"{n}. {when} — click to view (read-only)"):
                    _render_readonly_history_entry(
                        entry,
                        idx_label=f"Record #{n}",
                        widget_key_prefix="hist_tab_",
                    )
    
        if hist:
            bundle = json.dumps(
                {"records": [e.get("compliance", {}) for e in hist]},
                ensure_ascii=False,
                indent=2,
            )
            st.download_button(
                label="Download all compliance JSON (bundle)",
                data=bundle,
                file_name="compliance_history_bundle.json",
                mime="application/json",
                key="hist_tab_dl_compliance_bundle",
            )


if __name__ == "__main__":
    main()
