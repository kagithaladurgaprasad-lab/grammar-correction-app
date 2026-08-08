# 📝 English Grammar Correction System

An end-to-end **Natural Language Processing (NLP)** application that automatically corrects grammatical errors in English sentences using **Sequence-to-Sequence (Seq2Seq) Deep Learning models**.

The project implements and compares **LSTM, GRU, and Transformer-based architectures** and provides an interactive **Streamlit web application** for real-time grammar correction.

---

## 🚀 Live Demo

🔗 **Streamlit App:**
https://grammar-correction-app-4pt7ve4khxsk2cdb6uojwj.streamlit.app/

---

## 📌 Project Overview

Writing grammatically correct English is important for professional communication, education, documentation, and online content.

This project addresses the problem of **automatic grammatical error correction (GEC)** by treating grammar correction as a **sequence-to-sequence text generation problem**.

The system takes an incorrect English sentence as input and generates a corrected sentence as output.

### Example

**Input:**

```text
She go to school every day.
```

**Output:**

```text
She goes to school every day.
```

Another example:

**Input:**

```text
He don't likes cricket.
```

**Output:**

```text
He doesn't like cricket.
```

The primary objective is to correct grammatical errors while preserving the original meaning of the sentence.

---

## 🎯 Objectives

* Build an end-to-end English Grammar Correction system.
* Treat grammar correction as a Seq2Seq text generation problem.
* Implement **LSTM-based Encoder-Decoder architecture**.
* Implement **GRU-based Encoder-Decoder architecture**.
* Compare recurrent Seq2Seq models with a **Transformer-based architecture**.
* Implement text preprocessing and sequence preparation.
* Evaluate generated corrections using NLP evaluation techniques.
* Build an interactive Streamlit application.
* Deploy the application using Streamlit Community Cloud.

---

## 🧠 NLP Architecture

The project explores multiple sequence generation architectures.

```text
                    Input Sentence
                          │
                          ▼
                  Text Preprocessing
                          │
                          ▼
                     Tokenization
                          │
                          ▼
                  Sequence Encoding
                          │
                          ▼
                 ┌──────────────────┐
                 │                  │
                 ▼                  ▼
             LSTM Seq2Seq       GRU Seq2Seq
                 │                  │
                 └────────┬─────────┘
                          │
                          ▼
                    Transformer
                          │
                          ▼
                  Corrected Sentence
                          │
                          ▼
                  Streamlit Interface
```

---

# 🔹 1. Seq2Seq with LSTM

The first architecture uses an **Encoder-Decoder LSTM**.

### Encoder

The encoder processes the incorrect sentence and converts the input sequence into hidden-state representations.

```text
Incorrect Sentence
        ↓
Embedding
        ↓
LSTM Encoder
        ↓
Context Representation
```

### Decoder

The decoder uses the encoded information to generate the corrected sentence one token at a time.

```text
Context Representation
        ↓
LSTM Decoder
        ↓
Corrected Sentence
```

### Why LSTM?

LSTM is capable of learning long-term dependencies using:

* Forget Gate
* Input Gate
* Output Gate

This makes it more suitable than a traditional RNN for longer sequences.

---

# 🔹 2. Seq2Seq with GRU

The second architecture replaces LSTM with **GRU (Gated Recurrent Unit)**.

GRU uses:

* Update Gate
* Reset Gate

Compared with LSTM, GRU has fewer parameters and can provide faster training and inference while still capturing important sequence dependencies.

---

# 🔹 3. Transformer

The project also includes a Transformer-based approach.

Unlike recurrent architectures, Transformers use **Self-Attention** to model relationships between tokens.

```text
Input Tokens
     ↓
Embeddings
     ↓
Self-Attention
     ↓
Feed Forward Network
     ↓
Transformer Layers
     ↓
Output Sequence
```

The Transformer architecture provides stronger contextual modeling and allows parallel processing of sequence tokens.

---

# 🔄 Training Pipeline

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Duplicate Removal
     ↓
Text Normalization
     ↓
Tokenization
     ↓
Vocabulary Construction
     ↓
Integer Encoding
     ↓
Sequence Padding
     ↓
Train / Validation / Test Split
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Model Saving
     ↓
Streamlit Deployment
```

---

# 🧹 Text Preprocessing

The preprocessing pipeline includes:

* Cleaning raw text
* Removing unnecessary spaces
* Handling duplicate records
* Tokenization
* Vocabulary construction
* Integer encoding
* Sequence padding
* Preparing decoder input and target sequences
* Adding sequence boundary tokens where required

Special tokens such as:

```text
<start>
<end>
```

can be used to indicate the beginning and end of target sequences during Seq2Seq training and inference.

---

# 🎓 Teacher Forcing

During Seq2Seq training, **Teacher Forcing** is used to improve decoder learning.

Instead of feeding the decoder's previous prediction as the next input during training, the actual previous target token is provided.

```text
Actual Previous Token
        ↓
     Decoder
        ↓
Next Token Prediction
```

This helps the model learn the target sequence more efficiently.

---

# 🔍 Inference

During inference, the model generates the corrected sentence sequentially.

```text
Incorrect Sentence
       ↓
     Encoder
       ↓
Decoder receives <START>
       ↓
Predict next token
       ↓
Feed prediction back
       ↓
Predict next token
       ↓
Continue
       ↓
<END>
       ↓
Corrected Sentence
```

---

# 📊 Model Evaluation

The generated corrections can be evaluated by comparing the predicted sentence with the reference corrected sentence.

Evaluation focuses on:

* BLEU Score
* Sentence-level comparison
* Grammar correctness
* Fluency
* Preservation of original meaning
* Inference performance

### Why BLEU?

BLEU evaluates the similarity between generated text and reference text using n-gram overlap.

It is useful for comparing generated sequences against reference corrections, although it should not be treated as the only measure of grammatical correctness.

---

# 🖥️ Streamlit Application

The project includes an interactive Streamlit interface.

Users can:

1. Enter an English sentence containing grammatical errors.
2. Select the available model.
3. Run grammar correction.
4. View the corrected sentence.
5. Compare outputs from different architectures where supported.

### Application Workflow

```text
User Input
    ↓
Streamlit UI
    ↓
Preprocessing
    ↓
Selected NLP Model
    ↓
Sequence Generation
    ↓
Post-processing
    ↓
Corrected Sentence
```

---

# 🛠️ Technologies Used

### Programming Language

* Python

### NLP

* Natural Language Processing
* Text Preprocessing
* Tokenization
* Sequence-to-Sequence Learning
* Text Generation

### Deep Learning

* TensorFlow
* Keras
* LSTM
* GRU
* Transformer
* Encoder-Decoder Architecture
* Attention / Self-Attention

### Machine Learning

* Scikit-learn
* NumPy

### Deployment

* Streamlit
* Streamlit Community Cloud

### Version Control

* Git
* GitHub

---

# 📁 Project Structure

```text
grammar-correction-app/
│
├── app.py
│
├── models/
│   ├── lstm_model.h5
│   ├── ...
│
├── requirements.txt
│
├── .gitignore
│
├── .gitattributes
│
└── README.md
```

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/kagithaladurgaprasad-lab/grammar-correction-app.git
```

## 2. Navigate to the project directory

```bash
cd grammar-correction-app
```

## 3. Create a virtual environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
```

Activate:

```bash
source venv/bin/activate
```

---

# 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 📋 Requirements

The project uses libraries including:

```text
streamlit
tensorflow
torch
transformers
sentencepiece
numpy
scikit-learn
```

The exact versions are specified in `requirements.txt`.

---

# ☁️ Deployment

The application is deployed using **Streamlit Community Cloud**.

Deployment workflow:

```text
GitHub Repository
       ↓
Streamlit Community Cloud
       ↓
Install requirements
       ↓
Load Models
       ↓
Run app.py
       ↓
Public Web Application
```

---

# ⚠️ Deployment Considerations

Deep learning models can require significant memory during deployment.

The application therefore needs:

* Compatible Python version
* Compatible TensorFlow/Keras version
* Compatible Transformer/PyTorch versions
* Proper model files
* Appropriate dependency versions

Model serialization compatibility is particularly important when loading `.h5` models across different TensorFlow/Keras environments.

---

# 💡 Key Learning Outcomes

Through this project, I gained practical experience in:

* NLP text preprocessing
* Tokenization
* Vocabulary creation
* Sequence encoding
* Padding
* Encoder-Decoder architecture
* Seq2Seq learning
* LSTM
* GRU
* Attention mechanisms
* Transformer architecture
* Teacher Forcing
* Text generation
* NLP evaluation
* Model serialization
* Streamlit application development
* Cloud deployment
* Git and GitHub

---

# 🔮 Future Improvements

Possible improvements include:

* Improving correction quality with larger datasets.
* Supporting multiple languages.
* Adding Beam Search decoding.
* Improving unknown-word handling.
* Adding detailed error classification.
* Providing side-by-side comparison of incorrect and corrected text.
* Adding human evaluation alongside automatic metrics.
* Optimizing model inference for cloud deployment.
* Adding spelling-error correction in addition to grammatical correction.

---

# 👨‍💻 Author

**Kagithala Durga Prasad**

MCA | AI/ML | NLP | Deep Learning

📧 [kagithaladurgaprasad@gmail.com](mailto:kagithaladurgaprasad@gmail.com)

🔗 GitHub:
https://github.com/kagithaladurgaprasad-lab

🔗 LinkedIn:
https://linkedin.com/in/kagithala-durga-prasad-81251a295

---

## ⭐ Project Highlights

```text
✓ End-to-End NLP Project
✓ Grammar Error Correction
✓ Seq2Seq Architecture
✓ LSTM Encoder-Decoder
✓ GRU Encoder-Decoder
✓ Transformer Architecture
✓ Attention Mechanism
✓ Teacher Forcing
✓ Text Generation
✓ BLEU Evaluation
✓ Streamlit Application
✓ Cloud Deployment
```

If you find this project useful, consider giving the repository a ⭐.
