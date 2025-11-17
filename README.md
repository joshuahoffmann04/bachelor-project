# 🎓 RAG-Chatbot für Studienordnungen

Ein vollständiges **Retrieval-Augmented Generation (RAG)** System zur Beantwortung von Studierendenfragen zu Studienordnungen, Modulhandbüchern und Prüfungsregelungen der Philipps-Universität Marburg.

## 🎯 Kernziele

- **Faktentreue**: Absolute Korrektheit bei kritischen Informationen (ECTS, Fristen, Voraussetzungen)
- **Quellenattribution**: Jede Antwort mit konkreten Dokumentenstellen belegt
- **Abstaining**: Bei Unsicherheit "Ich weiß es nicht" statt Halluzinationen
- **Skalierbarkeit**: Architektur für andere Studiengänge erweiterbar

## 🏗️ Systemarchitektur

```
┌─────────────────┐
│  Datenquellen   │
│  - PDFs         │
│  - Webseiten    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │
│  - PDF Parser   │
│  - Web Scraper  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Chunking     │
│  - Semantic     │
│  - Hybrid       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Retrieval     │
│  - FAISS        │
│  - BM25         │
│  - RRF Fusion   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Generation    │
│  - OpenAI/      │
│    Claude/vLLM  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Antwort     │
│  + Quellen      │
└─────────────────┘
```

## 📁 Projektstruktur

```
rag-studienordnung/
├── data/
│   ├── raw/                 # Original PDFs
│   ├── processed/           # FAISS/BM25 Indices
│   ├── scraped/             # Gecachte Webseiten
│   └── chunks/              # Gespeicherte Chunks
├── src/
│   ├── preprocessing/       # PDF-Parsing & Document Models
│   ├── scraping/            # Web-Scraping & Caching
│   ├── chunking/            # Hybrid Chunking-Strategien
│   ├── retrieval/           # FAISS + BM25 Retrieval
│   ├── generation/          # LLM-Integration
│   ├── evaluation/          # RAGAS-Metriken
│   └── utils/               # Config & Logging
├── config/
│   ├── config.yaml          # Zentrale Konfiguration
│   └── prompts/             # System-Prompts
├── cli.py                   # Interaktive CLI
├── pipeline.py              # Indexing-Pipeline
├── requirements.txt         # Python Dependencies
└── README.md                # Diese Datei
```

## 🚀 Quick Start

### 1. Installation

```bash
# Repository klonen
cd rag-studienordnung

# Virtual Environment erstellen
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder: venv\Scripts\activate  # Windows

# Dependencies installieren
pip install -r requirements.txt
```

### 2. Konfiguration

```bash
# .env Datei erstellen
cp .env.example .env

# API-Keys eintragen
nano .env
```

Erforderliche API-Keys in `.env`:
```bash
OPENAI_API_KEY=sk-...
# oder
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Dokumente hinzufügen

```bash
# Prüfungsordnung (PDF) in data/raw/ ablegen
cp /pfad/zur/Pruefungsordnung_BSc_Inf_2024.pdf data/raw/

# URLs in config.yaml anpassen:
# - Modulhandbuch-URL
# - Veranstaltungskalender-URL
```

### 4. Indices erstellen

```bash
# Pipeline ausführen (scraped, parsed, chunked, indexed)
python pipeline.py build-index
```

Das wird:
- PDFs parsen
- Webseiten scrapen
- Dokumente chunken
- FAISS + BM25 Indices erstellen
- Alles in `data/processed/` speichern

### 5. Chatbot starten

```bash
# Interaktiver Modus
python cli.py interactive

# Einzelne Frage
python cli.py ask "Wie viele ECTS hat das Modul Algorithmen?"
```

## 💡 Beispielfragen

```
> Wie viele ECTS hat das Modul "Algorithmen und Datenstrukturen"?
> Welche Voraussetzungen gibt es für die Bachelorarbeit?
> Bis wann muss ich mich für Prüfungen anmelden?
> Welche Prüfungsform hat das Modul Datenbanken?
> Wie viele Wahlpflichtmodule muss ich belegen?
```

## 🔧 Technischer Stack

### Backend
- **Framework**: FastAPI (vorbereitet für API)
- **Vektor-DB**: FAISS (Facebook AI Similarity Search)
- **Keyword-Search**: BM25 (rank-bm25)
- **Embedding-Model**: `paraphrase-multilingual-mpnet-base-v2`

### Preprocessing
- **PDF-Parsing**: PyMuPDF, pdfplumber, Tabula
- **Web-Scraping**: BeautifulSoup4 + Requests
- **Chunking**: Hybrid (Semantic + Sliding Window)

### LLM Integration
- **Providers**: OpenAI, Claude, vLLM (lokal)
- **Abstraktionsschicht**: Einheitliche Interface-Klasse
- **Prompting**: Strikte Constraints gegen Halluzinationen

### Logging & Monitoring
- **Logging**: structlog (DSGVO-konform)
- **Metriken**: Latenz, Retrieval-Scores, Cache-Hits

## ⚙️ Konfiguration

Alle Einstellungen in `config/config.yaml`:

### Datenquellen konfigurieren

```yaml
sources:
  pruefungsordnung:
    type: pdf
    path: data/raw/Pruefungsordnung_BSc_Inf_2024.pdf
    enabled: true
    priority: high

  modulhandbuch:
    type: web
    url: "https://..."
    update_frequency: weekly
    enabled: true
```

### Chunking anpassen

```yaml
chunking:
  strategy: hybrid
  semantic:
    min_chunk_size: 100
    max_chunk_size: 512
  sliding_window:
    chunk_size: 512
    overlap: 50
```

### Retrieval-Gewichtung

```yaml
retrieval:
  hybrid:
    dense_weight: 0.7  # FAISS
    sparse_weight: 0.3  # BM25
    fusion_method: rrf
```

### LLM-Provider wählen

```yaml
llm:
  provider: openai  # oder: claude, vllm

  openai:
    model: gpt-4-turbo-preview
    temperature: 0.1  # Niedrig für Faktentreue
    max_tokens: 1000
```

## 📊 CLI-Befehle

### Pipeline

```bash
# Indices erstellen
python pipeline.py build-index

# Web-Cache leeren
python pipeline.py clear-cache

# Cache-Statistiken
python pipeline.py stats
```

### Chatbot

```bash
# Interaktiv
python cli.py interactive

# Einzelfrage
python cli.py ask "FRAGE"

# Custom Config
python cli.py --config custom_config.yaml interactive
```

### Interaktive Befehle

Innerhalb der CLI:
```
/help   - Zeige Hilfe
/stats  - Zeige System-Statistiken
/quit   - Beenden
```

## 🔒 DSGVO & Datenschutz

Das System ist DSGVO-konform designed:

- **Anonymisierung**: User-Queries werden gehasht
- **Keine Personendaten**: Nur aggregierte Statistiken
- **Logging**: Strukturiert, ohne sensible Daten
- **Cache**: Lokales Caching, keine Cloud-Uploads

Konfiguration in `config.yaml`:

```yaml
logging:
  privacy:
    anonymize_queries: true
    no_personal_data: true
    aggregate_only: true
```

## 🌐 Web-Scraping

### Ethisches Scraping

- **robots.txt**: Wird respektiert
- **Rate-Limiting**: 1 Request/Sekunde (konfigurierbar)
- **User-Agent**: Identifiziert sich korrekt
- **Caching**: Verhindert wiederholte Requests

### Cache-Verwaltung

```bash
# Cache-Statistiken
python pipeline.py stats

# Cache leeren
python pipeline.py clear-cache

# Manuelle Cache-Verwaltung
python -c "from src.scraping.cache_manager import CacheManager; \
           c = CacheManager('data/scraped'); \
           print(c.get_stats())"
```

## 🧪 Evaluierung (Optional)

Erweitere das System mit RAGAS:

```python
from src.evaluation.evaluator import RAGASEvaluator

evaluator = RAGASEvaluator(config)
results = evaluator.evaluate(test_dataset)

print(f"Context Relevance: {results['context_relevance']:.2f}")
print(f"Faithfulness: {results['faithfulness']:.2f}")
print(f"Answer Relevance: {results['answer_relevance']:.2f}")
```

## 🔧 Troubleshooting

### Problem: Indices nicht gefunden

```bash
# Lösung: Pipeline ausführen
python pipeline.py build-index
```

### Problem: OpenAI API Fehler

```bash
# API-Key prüfen
cat .env | grep OPENAI

# Neu setzen
export OPENAI_API_KEY=sk-...
```

### Problem: PDF-Parsing schlägt fehl

```bash
# Alternativen Parser probieren
# In config.yaml:
pdf_processing:
  parsers:
    - pdfplumber  # Statt pymupdf
    - tabula
```

### Problem: Web-Scraping schlägt fehl

```bash
# Cache leeren und neu versuchen
python pipeline.py clear-cache
python pipeline.py build-index

# Oder: Playwright für dynamische Seiten
# pip install playwright
# playwright install chromium
```

## 📈 Performance-Optimierung

### GPU-Beschleunigung

```yaml
# In config.yaml
embeddings:
  device: cuda  # Statt cpu

performance:
  gpu:
    enabled: true
    device_id: 0
```

### Batch-Processing

```yaml
embeddings:
  batch_size: 64  # Höher für GPU

performance:
  max_workers: 8  # Parallel processing
```

### Caching

```yaml
caching:
  embeddings:
    enabled: true
  query:
    enabled: true
    ttl: 3600
```

## 🚀 Erweiterungen

### Weitere Studiengänge hinzufügen

1. PDFs in `data/raw/` ablegen
2. In `config.yaml` neue Source eintragen:

```yaml
sources:
  pruefungsordnung_mathematik:
    type: pdf
    path: data/raw/PO_BSc_Mathematik.pdf
    enabled: true
```

3. Pipeline neu ausführen:
```bash
python pipeline.py build-index
```

### FastAPI-Server (vorbereitet)

```python
# api/main.py
from fastapi import FastAPI
from src.generation.rag_generator import RAGGenerator

app = FastAPI()

@app.post("/ask")
async def ask_question(question: str):
    result = await generator.generate(question)
    return result
```

```bash
uvicorn api.main:app --reload
```

### Custom Evaluation

```python
# Eigene Metriken in src/evaluation/
from src.evaluation.metrics import ECTSAccuracyMetric

metric = ECTSAccuracyMetric()
score = metric.evaluate(predictions, ground_truth)
```

## 📚 Weitere Ressourcen

- **RAGAS Framework**: https://docs.ragas.io/
- **FAISS Documentation**: https://github.com/facebookresearch/faiss
- **Sentence Transformers**: https://www.sbert.net/
- **LangChain**: https://python.langchain.com/

## 🤝 Beitragen

Contributions sind willkommen! Bitte:

1. Fork das Repository
2. Feature-Branch erstellen
3. Tests hinzufügen
4. Pull Request öffnen

## 📝 Lizenz

MIT License - siehe LICENSE Datei

## 👥 Autoren

- Entwickelt für die Philipps-Universität Marburg
- Bachelor-Projekt: RAG-Systeme für Hochschulverwaltung

## 🙏 Danksagungen

- **FAISS**: Facebook AI Research
- **Sentence Transformers**: UKP Lab
- **RAGAS**: Exploding Gradients
- **LangChain**: Harrison Chase

---

**Status**: ✅ Production-Ready
**Version**: 1.0.0
**Letztes Update**: 2024

Für Fragen oder Support: [GitHub Issues](https://github.com/...)
