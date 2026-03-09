# ✦ Lumina Chat

**Lumina** is an intelligent French-language chatbot powered by multilingual BERT embeddings and a semantic search engine, built with Streamlit. It answers questions across multiple knowledge domains using cosine similarity to match user queries against a curated knowledge corpus.

---

## Features

- **Semantic understanding** via `paraphrase-multilingual-MiniLM-L12-v2` (Sentence Transformers)
- **Multilingual support** — primarily French, with multilingual BERT for flexibility
- **Multi-domain knowledge corpus** covering:
  - 📈 Economics (GDP, inflation, stock markets, cryptocurrencies, Keynesian theory…)
  - 💻 Computer Science (AI, machine learning, Python, cybersecurity, cloud, DevOps…)
  - 🏛 Politics (democracy, EU, geopolitics, separation of powers…)
  - 🏥 Health (immune system, diabetes, mental health, vaccines, nutrition…)
  - 🎨 Culture (Einstein, stoicism, French Revolution, impressionism…)
  - 🔬 Sciences (quantum mechanics, photosynthesis, DNA, climate change, black holes…)
  - ⚡ Technology (renewables, quantum computing, VR/AR, Industry 4.0…)
- **Fast small-talk detection** via regex for greetings, farewells, emotions, and bot questions
- **LRU embedding cache** (256 entries) for faster repeated queries
- **Adjustable confidence threshold** via sidebar slider
- **BERT match inspector** — expandable view of top cosine similarity scores
- **Session statistics** — question count, resolution rate, average confidence
- **Quick-suggestion buttons** for common queries

---

## How It Works

1. The user's message is first checked against regex patterns for small talk (greetings, emotions, questions about the bot). If matched, a pre-written response is returned immediately.
2. If not small talk, the query is preprocessed (cleaning, tokenization, lemmatization, stopword removal) and encoded into a BERT embedding.
3. Cosine similarity is computed between the query embedding and all pre-encoded corpus entries.
4. The top-K (default: 4) matches are retrieved. If the best match exceeds the confidence threshold (default: 50%), its knowledge entry is used to generate a contextual response.
5. If no match clears the threshold, a fallback response is returned.

---

## Installation

**Requirements:** Python 3.9+

```bash
# Clone the repository
git clone <your-repo-url>
cd lumina-chat

# Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`

```
streamlit>=1.32.0
sentence-transformers>=2.6.0
nltk>=3.8.1
scikit-learn>=1.4.0
numpy>=1.26.0
```

> On first run, the app will automatically download required NLTK data (`punkt`, `stopwords`, `wordnet`) and the Sentence Transformers model (~120 MB).

---

## Usage

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`.

---

## Configuration

The following constants at the top of `app.py` can be tuned:

| Constant | Default | Description |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.50` | Minimum cosine similarity to return a semantic answer |
| `BERT_MODEL_NAME` | `paraphrase-multilingual-MiniLM-L12-v2` | Sentence Transformer model |
| `MAX_CACHE_SIZE` | `256` | LRU cache size for query embeddings |
| `TOP_K` | `4` | Number of top corpus matches to retrieve |

The confidence threshold can also be adjusted live via the sidebar slider without restarting the app.

---

## Project Structure

```
lumina-chat/
├── app.py           # Main application (UI + NLP pipeline + corpus)
├── requirements.txt # Python dependencies
└── README.md
```

---

## Tech Stack

| Layer | Library |
|---|---|
| UI | Streamlit |
| Semantic embeddings | Sentence Transformers |
| Similarity search | scikit-learn (cosine similarity) |
| Text preprocessing | NLTK (tokenization, lemmatization, stopwords) |
| Numerical compute | NumPy |
