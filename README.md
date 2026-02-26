# 🚀 Intelligent Search: Hybrid Property Match Engine

A high-performance search engine built with **Python**, **Sentence-Transformers**, and **Matrix Math**. This project solves "search fatigue" by combining strict structured data with AI-driven semantic intent.

---

## 🧠 The Core Logic: 80% Match Engine

Unlike traditional search tools that use "binary" results (you either match or you don't), this engine calculates a **weighted relevance score** for every listing to find the "perfect" home:

1. **Hard Filters (50% Weight):** Strict numeric validation for `Price` and `Bedroom` count. If a listing fails these, it is heavily penalized, ensuring budget and space requirements are met first.
2. **Semantic Intelligence (50% Weight):** Uses the `all-MiniLM-L6-v2` model to understand natural language intent. It can find "cozy," "modern," or "family-friendly" homes even if those exact keywords aren't in the database.

---

## 🛠️ Tech Stack & Performance

| Layer            | Technology                              |
| ---------------- | --------------------------------------- |
| **Engine**       | Python 3.12 + Pandas                    |
| **AI/NLP**       | Sentence-Transformers (PyTorch backend) |
| **Optimization** | Vectorized Matrix Multiplication        |

> The engine pre-calculates embeddings for the entire dataset once, allowing for **sub-10ms search latency** across thousands of records.

---

## 📂 Project Structure

```
.
├── MatchEngine.py              # The core class-based engine
├── apartments_for_rent.csv     # Dataset source (semicolon-separated)
├── requirements.txt            # Project dependencies
└── .gitignore                  # Prevents large datasets and venvs from cluttering the repo
```

---

## 🚀 Getting Started

### Prerequisites

Ensure you are using a virtual environment to keep dependencies isolated:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Installation

Install all dependencies from the requirements file:

```bash
pip install -r requirements.txt
```

### Usage

Run the interactive CLI to test different lifestyle queries:

```bash
python3 MatchEngine.py
```

## 📜 License

MIT — Created by **Kamogelo Mmopane**
