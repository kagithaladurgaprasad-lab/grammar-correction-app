<<<<<<< HEAD
# GrammarAI — LSTM vs GRU vs Transformer Grammar Correction

A Streamlit app that lets you **slide between three grammar-correction models**
trained/used in your seq2seq project and compare their output live, with
word-level diff highlighting.

- 🔵 **LSTM** — custom encoder–decoder seq2seq, trained from scratch in Colab
- 🟡 **GRU** — custom encoder–decoder seq2seq, trained from scratch in Colab
- 🌸 **Transformer** — pretrained `pszemraj/flan-t5-large-grammar-synthesis` from Hugging Face

## 1. Where the model files come from

The **Transformer** downloads automatically from the Hugging Face Hub the first
time the app runs — you don't need to upload anything for it.

The **LSTM** and **GRU** models were trained in your Colab notebook and only
exist there — you must export them and add them to this repo yourself.

### Export from Colab

At the end of your notebook (after training both models), run:

```python
import json
from google.colab import files

lstm_model.save("lstm_model.h5")
gru_model.save("gru_model.h5")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

config = {"MAX_LEN": MAX_LEN, "EMB_DIM": EMB_DIM, "HID_DIM": HID_DIM, "VOCAB_SIZE": VOCAB_SIZE}
with open("config.json", "w") as f:
    json.dump(config, f)

for fname in ["lstm_model.h5", "gru_model.h5", "tokenizer.pkl", "config.json"]:
    files.download(fname)
```

This downloads 4 files to your computer. Put them in a `models/` folder next
to `app.py`:

```
grammar_app/
├── app.py
├── requirements.txt
├── README.md
└── models/
    ├── lstm_model.h5
    ├── gru_model.h5
    ├── tokenizer.pkl
    └── config.json
```

> **Why not save separate inference models?** The app rebuilds the step-by-step
> encoder/decoder inference models from the single saved training model's layers
> at load time, so you only need to save one `.h5` file per architecture.

## 2. Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 3. Deploy — Streamlit Community Cloud

1. Push the whole `grammar_app/` folder (including `models/`) to a **public
   GitHub repo**.
2. **Check file sizes** — GitHub blocks files over 100MB:
   ```bash
   ls -lh models/
   ```
   If `lstm_model.h5` or `gru_model.h5` is close to/over 100MB, use
   [Git LFS](https://git-lfs.github.com/):
   ```bash
   git lfs install
   git lfs track "*.h5"
   git add .gitattributes models/*.h5
   git commit -m "add models via LFS"
   git push
   ```
3. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
4. Click **New app**, pick your repo/branch, set main file to `app.py`, deploy.

**Free tier is CPU-only with limited RAM.** Loading all three models (especially
`flan-t5-large`, ~780M params) can be slow or hit memory limits. The app only
loads the model you select (lazy loading + caching), so this mostly affects
whichever model is picked first. If the Transformer is too heavy for the free
tier, open `app.py` and change:
```python
HF_MODEL_NAME = "pszemraj/flan-t5-large-grammar-synthesis"
```
to the smaller:
```python
HF_MODEL_NAME = "vennify/t5-base-grammar-correction"
```

## 4. Alternative — Hugging Face Spaces (supports optional GPU)

1. Create a Space at [huggingface.co/new-space](https://huggingface.co/new-space)
   with the **Streamlit** SDK.
2. Push/upload `app.py`, `requirements.txt`, and the `models/` folder
   (Spaces has native Git LFS support, better suited for larger `.h5` files).
3. Optionally enable a GPU tier in Space settings.

## Troubleshooting

- **"Couldn't find model files" error in the app** → double-check the `models/`
  folder is in the repo and the four filenames match exactly
  (`lstm_model.h5`, `gru_model.h5`, `tokenizer.pkl`, `config.json`).
- **LSTM/GRU output looks empty or garbled** → the model may need more training
  epochs, or the sentence is longer than `MAX_LEN` tokens (check `config.json`).
- **App is slow on first correction per model** → that's the model loading
  (cached after first use per session, via `st.cache_resource`).
=======
# grammar-correction-app
>>>>>>> c6bc590b5b71d9ae904db27c119ac106031dff8b
