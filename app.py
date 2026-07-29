import difflib
import json
import os
import pickle
import time

import numpy as np
import streamlit as st
import torch
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="GrammarAI — Compare LSTM · GRU · Transformer",
    page_icon="✒️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS_DIR = "models"
HF_MODEL_NAME = "pszemraj/flan-t5-large-grammar-synthesis"

# ----------------------------------------------------------------------------
# Custom CSS
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

        .stApp {
            background: radial-gradient(circle at 10% 0%, #1b1035 0%, #0d0b1f 45%, #0a0912 100%);
            color: #f2f0fb;
        }

        #MainMenu, header, footer {visibility: hidden;}

        .hero {
            padding: 2.2rem 2.5rem 1.6rem 2.5rem;
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(124,58,237,0.35), rgba(236,72,153,0.25));
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 1.4rem;
            box-shadow: 0 20px 60px rgba(90, 40, 180, 0.25);
        }

        .hero h1 {
            font-family: 'Sora', sans-serif;
            font-weight: 800;
            font-size: 2.6rem;
            margin-bottom: 0.2rem;
            background: linear-gradient(90deg, #f472b6, #a78bfa 45%, #60a5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero p { font-size: 1.05rem; color: #cbd0e6; margin-top: 0; max-width: 680px; }

        .badge-row { margin-top: 0.9rem; display: flex; gap: 0.5rem; flex-wrap: wrap; }

        .badge {
            display: inline-block;
            padding: 0.35rem 0.85rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.12);
            color: #e5e1ff;
        }

        .model-picker {
            padding: 1.3rem 1.6rem;
            border-radius: 20px;
            background: rgba(255,255,255,0.045);
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 1.3rem;
        }

        .model-picker h3 {
            font-family: 'Sora', sans-serif;
            font-size: 1rem;
            margin-top: 0;
            margin-bottom: 0.7rem;
            color: #f5f3ff;
        }

        .model-tag {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 700;
            margin-top: 0.6rem;
        }
        .tag-lstm { background: rgba(96,165,250,0.18); color: #93c5fd; border: 1px solid rgba(96,165,250,0.4); }
        .tag-gru { background: rgba(251,191,36,0.18); color: #fcd34d; border: 1px solid rgba(251,191,36,0.4); }
        .tag-transformer { background: rgba(244,114,182,0.18); color: #f9a8d4; border: 1px solid rgba(244,114,182,0.4); }

        .card {
            background: rgba(255,255,255,0.045);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 20px;
            padding: 1.5rem 1.6rem;
            backdrop-filter: blur(6px);
            margin-bottom: 1.2rem;
        }
        .card h3 { font-family: 'Sora', sans-serif; font-size: 1.05rem; font-weight: 700; margin-top: 0; color: #f5f3ff; }

        .result-box {
            background: rgba(16, 185, 129, 0.08);
            border: 1px solid rgba(16, 185, 129, 0.35);
            border-radius: 16px;
            padding: 1.1rem 1.3rem;
            font-size: 1.05rem;
            line-height: 1.65;
            color: #eafff5;
        }

        .diff-add { background: rgba(16, 185, 129, 0.28); color: #baffdf; padding: 0.05rem 0.25rem; border-radius: 6px; font-weight: 600; }
        .diff-del { background: rgba(244, 63, 94, 0.28); color: #ffc4d1; padding: 0.05rem 0.25rem; border-radius: 6px; text-decoration: line-through; }

        .metric-pill {
            display: inline-flex; flex-direction: column; align-items: center; justify-content: center;
            padding: 0.9rem 1.2rem; border-radius: 16px;
            background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08);
            min-width: 110px; margin-right: 0.7rem;
        }
        .metric-pill .num { font-family: 'Sora', sans-serif; font-size: 1.5rem; font-weight: 800; color: #c4b5fd; }
        .metric-pill .label { font-size: 0.72rem; color: #a8adc9; text-transform: uppercase; letter-spacing: 0.06em; }

        div.stButton > button {
            background: linear-gradient(90deg, #7c3aed, #ec4899);
            color: white; border: none; border-radius: 14px;
            padding: 0.65rem 1.6rem; font-weight: 700; font-size: 1rem;
            box-shadow: 0 8px 24px rgba(124, 58, 237, 0.35);
            transition: transform 0.15s ease;
        }
        div.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 12px 28px rgba(236, 72, 153, 0.4); }

        .stTextArea textarea {
            background: rgba(255,255,255,0.04) !important;
            color: #f2f0fb !important;
            border-radius: 14px !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            font-size: 1.02rem !important;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #150c2e, #0b0a18);
            border-right: 1px solid rgba(255,255,255,0.06);
        }

        .history-item {
            padding: 0.7rem 0.9rem; border-radius: 12px;
            background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.07);
            margin-bottom: 0.55rem; font-size: 0.85rem;
        }
        .history-item .orig { color: #ff9fb3; text-decoration: line-through; }
        .history-item .fixed { color: #8ff5c4; }
        .history-item .modeltag { color: #93c5fd; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Loaders — one cached function per model, so only the selected one loads
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_tokenizer_and_config():
    with open(os.path.join(MODELS_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "config.json")) as f:
        config = json.load(f)
    return tokenizer, config


def _rebuild_inference_models(full_model, rnn_type, hid_dim):
    """Reconstruct step-by-step encoder/decoder inference models from the
    saved training model's layers (weights are reused, not retrained)."""
    layer_suffix = "lstm" if rnn_type == "lstm" else "gru"

    encoder_input_tensor = full_model.get_layer("encoder_input").output
    encoder_rnn_layer = full_model.get_layer(f"encoder_{layer_suffix}")
    encoder_layer_outputs = encoder_rnn_layer.output  # [seq_out, state_h, (state_c)]
    encoder_states = list(encoder_layer_outputs[1:])
    encoder_model = Model(encoder_input_tensor, encoder_states)

    decoder_embedding_layer = full_model.get_layer("decoder_embedding")
    decoder_rnn_layer = full_model.get_layer(f"decoder_{layer_suffix}")
    decoder_dense_layer = full_model.get_layer("decoder_dense")

    dec_input_single = Input(shape=(1,))
    dec_emb_single = decoder_embedding_layer(dec_input_single)

    if rnn_type == "lstm":
        dec_state_h = Input(shape=(hid_dim,))
        dec_state_c = Input(shape=(hid_dim,))
        dec_out, out_h, out_c = decoder_rnn_layer(dec_emb_single, initial_state=[dec_state_h, dec_state_c])
        dec_states_inputs = [dec_state_h, dec_state_c]
        dec_states_outputs = [out_h, out_c]
    else:
        dec_state_h = Input(shape=(hid_dim,))
        dec_out, out_h = decoder_rnn_layer(dec_emb_single, initial_state=[dec_state_h])
        dec_states_inputs = [dec_state_h]
        dec_states_outputs = [out_h]

    dec_out = decoder_dense_layer(dec_out)
    decoder_model = Model([dec_input_single] + dec_states_inputs, [dec_out] + dec_states_outputs)

    return encoder_model, decoder_model


@st.cache_resource(show_spinner=False)
def load_lstm():
    _, config = load_tokenizer_and_config()
    full_model = load_model(os.path.join(MODELS_DIR, "lstm_model.h5"))
    return _rebuild_inference_models(full_model, "lstm", config["HID_DIM"])


@st.cache_resource(show_spinner=False)
def load_gru():
    _, config = load_tokenizer_and_config()
    full_model = load_model(os.path.join(MODELS_DIR, "gru_model.h5"))
    return _rebuild_inference_models(full_model, "gru", config["HID_DIM"])


@st.cache_resource(show_spinner=False)
def load_transformer():
    tok = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = mdl.to(device)
    mdl.eval()
    return tok, mdl, device


# ----------------------------------------------------------------------------
# Inference functions
# ----------------------------------------------------------------------------
def clean_text(text):
    import re
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def greedy_decode_keras(encoder_model, decoder_model, tokenizer, config, sentence, rnn_type):
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    max_len = config["MAX_LEN"]
    seq = tokenizer.texts_to_sequences([clean_text(sentence)])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")

    states = encoder_model.predict(padded, verbose=0)
    states = list(states) if rnn_type == "lstm" else [states]

    target_seq = np.array([[tokenizer.word_index.get("sos", 1)]])
    decoded_words = []

    for _ in range(max_len):
        outputs = decoder_model.predict([target_seq] + states, verbose=0)
        output_tokens, states = outputs[0], outputs[1:]
        sampled_idx = int(np.argmax(output_tokens[0, -1, :]))
        sampled_word = tokenizer.index_word.get(sampled_idx, "")

        if sampled_word in ("eos", ""):
            break
        decoded_words.append(sampled_word)
        target_seq = np.array([[sampled_idx]])

    return " ".join(decoded_words) if decoded_words else "(model produced no output — try a shorter sentence)"


def correct_with_transformer(text, tok, mdl, device, max_length=128, num_beams=5):
    inputs = tok(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        output_ids = mdl.generate(**inputs, max_length=max_length, num_beams=num_beams, early_stopping=True)
    return tok.decode(output_ids[0], skip_special_tokens=True)


def render_diff_html(original, corrected):
    orig_words = original.split()
    corr_words = corrected.split()
    matcher = difflib.SequenceMatcher(None, orig_words, corr_words)
    html_parts = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            html_parts.append(" ".join(corr_words[j1:j2]))
        elif tag == "replace":
            html_parts.append(f'<span class="diff-del">{" ".join(orig_words[i1:i2])}</span>')
            html_parts.append(f'<span class="diff-add">{" ".join(corr_words[j1:j2])}</span>')
        elif tag == "delete":
            html_parts.append(f'<span class="diff-del">{" ".join(orig_words[i1:i2])}</span>')
        elif tag == "insert":
            html_parts.append(f'<span class="diff-add">{" ".join(corr_words[j1:j2])}</span>')
    return " ".join(html_parts)


def count_edits(original, corrected):
    matcher = difflib.SequenceMatcher(None, original.split(), corrected.split())
    return sum(1 for tag, *_ in matcher.get_opcodes() if tag != "equal")


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

EXAMPLES = [
    "She dont like to eat vegetables but she like fruits.",
    "Me and him was going to the store yesterday.",
    "Their are many reason why this happen.",
    "He don't never listen to nobody.",
    "The team are excited for there upcoming match tommorow.",
]

MODEL_META = {
    "LSTM": {"tag_class": "tag-lstm", "emoji": "🔵", "desc": "Custom encoder–decoder LSTM, trained from scratch"},
    "GRU": {"tag_class": "tag-gru", "emoji": "🟡", "desc": "Custom encoder–decoder GRU, trained from scratch"},
    "Transformer": {"tag_class": "tag-transformer", "emoji": "🌸", "desc": f"Pretrained: {HF_MODEL_NAME}"},
}

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ About this app")
    st.markdown(
        "Compare three grammar-correction approaches from the same project: "
        "two **custom seq2seq models trained from scratch** (LSTM, GRU), and a "
        "**pretrained Transformer** as an upper-bound baseline."
    )
    st.markdown("---")
    st.markdown("### 💡 Try an example")
    for ex in EXAMPLES:
        if st.button(ex, key=f"ex_{hash(ex)}", use_container_width=True):
            st.session_state.input_text = ex
    st.markdown("---")
    st.markdown("### 🕘 Recent corrections")
    if st.session_state.history:
        for item in reversed(st.session_state.history[-5:]):
            st.markdown(
                f"""<div class="history-item">
                    <div class="modeltag">{item['model']}</div>
                    <div class="orig">{item['original']}</div>
                    <div class="fixed">{item['corrected']}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        if st.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("Your corrected sentences will show up here.")

# ----------------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>✒️ GrammarAI</h1>
        <p>Slide between three grammar-correction models — LSTM, GRU, and a pretrained
        Transformer — and see how each one rewrites your sentence, side by side with
        a word-level diff.</p>
        <div class="badge-row">
            <span class="badge">🔵 LSTM seq2seq</span>
            <span class="badge">🟡 GRU seq2seq</span>
            <span class="badge">🌸 Pretrained Transformer</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Model picker (slider)
# ----------------------------------------------------------------------------
st.markdown('<div class="model-picker"><h3>🎚️ Choose a model</h3>', unsafe_allow_html=True)
selected_model = st.select_slider(
    label="model_picker",
    options=["LSTM", "GRU", "Transformer"],
    value="Transformer",
    label_visibility="collapsed",
)
meta = MODEL_META[selected_model]
st.markdown(
    f'<span class="model-tag {meta["tag_class"]}">{meta["emoji"]} {selected_model} — {meta["desc"]}</span>',
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Main layout
# ----------------------------------------------------------------------------
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="card"><h3>✍️ Your text</h3>', unsafe_allow_html=True)
    text_input = st.text_area(
        label="input",
        value=st.session_state.input_text,
        height=220,
        placeholder="Type or paste a sentence with grammar mistakes here...",
        label_visibility="collapsed",
        key="main_input",
    )

    if selected_model == "Transformer":
        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_a:
            num_beams = st.slider("Beam width", min_value=1, max_value=8, value=5)
        with col_b:
            max_len_ui = st.slider("Max output length", min_value=32, max_value=256, value=128, step=16)
        with col_c:
            st.write("")
            st.write("")
            run = st.button("✨ Correct grammar", use_container_width=True)
    else:
        st.caption(f"{selected_model} uses greedy decoding — no extra settings needed.")
        run = st.button("✨ Correct grammar", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card"><h3>✅ Corrected result</h3>', unsafe_allow_html=True)

    if run and text_input.strip():
        try:
            start = time.time()
            with st.spinner(f"Loading {selected_model} and polishing your sentence..."):
                if selected_model == "LSTM":
                    tokenizer, config = load_tokenizer_and_config()
                    encoder_model, decoder_model = load_lstm()
                    corrected = greedy_decode_keras(
                        encoder_model, decoder_model, tokenizer, config, text_input.strip(), "lstm"
                    )
                elif selected_model == "GRU":
                    tokenizer, config = load_tokenizer_and_config()
                    encoder_model, decoder_model = load_gru()
                    corrected = greedy_decode_keras(
                        encoder_model, decoder_model, tokenizer, config, text_input.strip(), "gru"
                    )
                else:
                    tok, mdl, device = load_transformer()
                    corrected = correct_with_transformer(
                        text_input.strip(), tok, mdl, device, max_length=max_len_ui, num_beams=num_beams
                    )
            elapsed = time.time() - start

            st.markdown(f'<div class="result-box">{corrected}</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Word-level changes:**")
            st.markdown(
                f'<div class="result-box">{render_diff_html(text_input.strip(), corrected)}</div>',
                unsafe_allow_html=True,
            )

            n_edits = count_edits(text_input.strip(), corrected)
            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(
                    f'<div class="metric-pill"><div class="num">{n_edits}</div><div class="label">Edits made</div></div>',
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f'<div class="metric-pill"><div class="num">{elapsed:.2f}s</div><div class="label">Inference time</div></div>',
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f'<div class="metric-pill"><div class="num">{len(corrected.split())}</div><div class="label">Words out</div></div>',
                    unsafe_allow_html=True,
                )

            st.download_button(
                "⬇ Download corrected text",
                data=corrected,
                file_name=f"corrected_{selected_model.lower()}.txt",
                mime="text/plain",
                use_container_width=True,
            )

            st.session_state.history.append(
                {"model": selected_model, "original": text_input.strip(), "corrected": corrected}
            )

        except FileNotFoundError as e:
            st.error(
                f"Couldn't find model files for **{selected_model}**. "
                f"Make sure `models/lstm_model.h5`, `models/gru_model.h5`, "
                f"`models/tokenizer.pkl`, and `models/config.json` are all present in your repo.\n\n"
                f"Missing: `{e.filename}`"
            )

    elif run and not text_input.strip():
        st.warning("Type something first — the text box is empty.")
    else:
        st.caption("Your corrected sentence will appear here once you click **Correct grammar**.")

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; color:#7d81a0; font-size:0.8rem; margin-top:2rem;">
        Built with Streamlit · TensorFlow/Keras · Hugging Face Transformers — Grammar Correction Seq2Seq Project
    </div>
    """,
    unsafe_allow_html=True,
)
