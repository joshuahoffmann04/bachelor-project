# 🔧 RAG System Optimierungs-Guide

Dein RAG-System läuft, aber die Ergebnisse sind schlecht? Dieser Guide hilft dir systematisch zu debuggen und zu optimieren.

## 🔍 Schritt 1: Diagnose - Was ist das Problem?

### A) Schnell-Diagnose

```bash
# Vollständiger Diagnose-Report
python debug.py full-report

# Nur Chunks analysieren
python debug.py analyze-chunks

# Retrieval testen
python debug.py test-retrieval "Wie viele ECTS hat Algorithmen?"

# Generation testen
python debug.py test-generation "Wie viele ECTS hat Algorithmen?"

# Test-Suite laufen lassen
python debug.py test-suite
```

### B) Typische Probleme identifizieren

#### Problem 1: **Retrieval findet falsche Chunks**

**Symptome:**
- Antworten sind komplett off-topic
- Quellen passen nicht zur Frage
- Confidence-Scores sind niedrig (<0.3)

**Test:**
```bash
python debug.py test-retrieval "Wie viele ECTS hat Algorithmen?"
```

**Was du sehen solltest:**
- Top-3 Ergebnisse sollten relevante Modulinformationen enthalten
- Scores sollten >0.5 sein bei guten Matches

**Falls nicht → Gehe zu Fix A**

#### Problem 2: **Chunking ist suboptimal**

**Symptome:**
- Chunks sind zu groß/klein
- Tabellen sind zerrissen
- Wichtige Informationen fehlen

**Test:**
```bash
python debug.py analyze-chunks
```

**Was du sehen solltest:**
- Avg Tokens/Chunk: 200-512 (optimal)
- Tabellen als separate Chunks (Type: table)
- Gute Mix aus paragraph, table, list

**Falls nicht → Gehe zu Fix B**

#### Problem 3: **LLM halluziniert oder abstained zu oft**

**Symptome:**
- Antworten enthalten erfundene Infos
- Oder: System sagt zu oft "Ich weiß es nicht"

**Test:**
```bash
python debug.py test-suite
```

**Was du sehen solltest:**
- Bei klaren Fragen (ECTS): Confidence >0.8
- Bei Out-of-Scope: Abstaining
- Keine erfundenen Zahlen

**Falls nicht → Gehe zu Fix C**

#### Problem 4: **Performance-Probleme**

**Symptome:**
- Retrieval >1 Sekunde
- Generation >5 Sekunden
- System hängt

**Falls ja → Gehe zu Fix D**

---

## 🛠️ Schritt 2: Fixes

### Fix A: Retrieval verbessern

#### A1: Embedding-Modell wechseln

Das Standard-Modell ist gut für Multilingual, aber vielleicht nicht optimal für deutsche Fachtexte.

**Option 1: Deutsche-spezifisches Modell**
```yaml
# config/config.yaml
embeddings:
  model: deutsche-telekom/gbert-large-paraphrase-cosine
  # Oder: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

**Option 2: Größeres Modell**
```yaml
embeddings:
  model: sentence-transformers/paraphrase-multilingual-mpnet-base-v2  # CURRENT
  # Upgrade zu:
  model: intfloat/multilingual-e5-large
```

Nach Änderung:
```bash
python pipeline.py build-index  # Neu indexieren!
```

#### A2: Retrieval-Gewichtung anpassen

Wenn BM25 (Keyword-Search) bessere Ergebnisse liefert als FAISS:

```yaml
# config/config.yaml
retrieval:
  hybrid:
    dense_weight: 0.5  # Runter von 0.7
    sparse_weight: 0.5  # Hoch von 0.3
```

Für exakte Keyword-Matches (Modulcodes, §-Referenzen):
```yaml
retrieval:
  hybrid:
    dense_weight: 0.3
    sparse_weight: 0.7  # BM25 dominiert
```

#### A3: Top-K erhöhen

Mehr Chunks abrufen:
```yaml
retrieval:
  dense:
    top_k: 20  # Statt 10
  sparse:
    top_k: 20
  final_top_k: 10  # An LLM senden (nicht zu hoch!)
```

#### A4: Similarity-Threshold senken

Falls zu wenige Ergebnisse:
```yaml
retrieval:
  dense:
    similarity_threshold: 0.3  # Runter von 0.5
```

**Testen:**
```bash
python debug.py test-retrieval "Deine Testfrage"
```

### Fix B: Chunking optimieren

#### B1: Chunk-Größe anpassen

**Chunks zu groß** (>1000 Tokens):
```yaml
chunking:
  semantic:
    max_chunk_size: 384  # Runter von 512
  sliding_window:
    chunk_size: 384
    overlap: 64  # Mehr Overlap
```

**Chunks zu klein** (<100 Tokens):
```yaml
chunking:
  semantic:
    min_chunk_size: 150  # Hoch von 100
    max_chunk_size: 768
```

#### B2: Strategie ändern

Wenn Semantic Chunking Tabellen zerreißt:
```yaml
chunking:
  strategy: hybrid  # Oder: semantic
  special_handling:
    tables:
      split: false  # WICHTIG!
      max_size: 2048  # Größer für lange Tabellen
```

Nur Sliding Window (einfacher):
```yaml
chunking:
  strategy: sliding_window
  sliding_window:
    chunk_size: 512
    overlap: 100  # 20% Overlap
```

#### B3: Overlap erhöhen

Damit wichtiger Kontext nicht verloren geht:
```yaml
chunking:
  sliding_window:
    chunk_size: 512
    overlap: 128  # 25% Overlap (hoch!)
```

**Testen:**
```bash
python pipeline.py build-index  # Neu chunken!
python debug.py analyze-chunks   # Prüfen
```

### Fix C: LLM-Generation verbessern

#### C1: Prompts optimieren

**Problem: Zu viele Halluzinationen**

Editiere `config/prompts/system_prompt.txt`:
```txt
Du bist ein ULTRA-PRÄZISER Assistent.

KRITISCH WICHTIG:
1. Wenn der Kontext die Antwort nicht EXPLIZIT enthält → ABSTAIN
2. KEINE Vermutungen, KEINE Interpretationen
3. Bei Zahlen (ECTS, Fristen): NUR exakte Werte aus Kontext
4. Lieber 10x "Ich weiß es nicht" als 1x falsch

[... Rest des Prompts ...]
```

**Problem: Zu viel Abstaining**

```yaml
# config/config.yaml
prompts:
  abstaining:
    threshold: 0.5  # Runter von 0.6
```

Oder Prompt anpassen:
```txt
Bei Konfidenz >0.5: Antworte mit Hinweis auf Unsicherheit
Bei Konfidenz <0.5: Vollständig abstain
```

#### C2: Temperatur anpassen

**Für mehr Faktentreue:**
```yaml
llm:
  openai:
    temperature: 0.0  # Runter von 0.1 (deterministisch!)
```

**Für etwas kreativere Antworten:**
```yaml
llm:
  openai:
    temperature: 0.3
```

#### C3: Kontext-Format verbessern

Editiere `config/prompts/user_prompt_template.txt`:

**Aktuell:**
```txt
KONTEXT: {context}
```

**Besser:**
```txt
DOKUMENTE:

{context}

WICHTIG: Antworte NUR basierend auf den OBEN gezeigten Dokumenten.
Zitiere IMMER die [Nummer] des Dokuments in deiner Antwort.
```

#### C4: Provider wechseln

Falls OpenAI GPT-4 halluziniert:

```yaml
llm:
  provider: claude  # Statt openai

  claude:
    model: claude-3-opus-20240229  # Sehr faktentreu
    temperature: 0.0
```

Claude ist oft präziser bei faktischen Fragen.

**Testen:**
```bash
python debug.py test-generation "Wie viele ECTS hat Algorithmen?"
```

### Fix D: Performance optimieren

#### D1: GPU aktivieren

Falls du eine GPU hast:
```yaml
embeddings:
  device: cuda  # Statt cpu

performance:
  gpu:
    enabled: true
```

#### D2: Batch-Size erhöhen

```yaml
embeddings:
  batch_size: 64  # Hoch von 32 (mit GPU)
```

#### D3: Caching aktivieren

```yaml
caching:
  enabled: true
  query:
    enabled: true
    ttl: 3600
    max_size: 1000
```

#### D4: Index optimieren

Für >10,000 Chunks:
```yaml
retrieval:
  dense:
    index_type: ivf  # Statt flatl2
```

Dann re-index:
```bash
python pipeline.py build-index
```

---

## 🎯 Schritt 3: Iteratives Tuning

### 3.1 Test-Driven Optimization

1. **Erstelle eine Testfragen-Datei:**

```bash
cat > test_questions.txt << 'EOF'
Wie viele ECTS hat das Modul Algorithmen und Datenstrukturen?
Welche Voraussetzungen gibt es für die Bachelorarbeit?
Bis wann muss ich mich für Prüfungen anmelden?
Welche Prüfungsform hat Datenbanken?
Wie viele Wahlpflichtmodule muss ich belegen?
EOF
```

2. **Automatisches Testen:**

```python
# test_runner.py
import asyncio
from src.generation.rag_generator import RAGGenerator
from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.config_loader import load_config

async def test_all():
    config = load_config()
    retriever = HybridRetriever(config)
    retriever.load_indices(...)
    generator = RAGGenerator(retriever, config=config)

    with open('test_questions.txt') as f:
        questions = f.readlines()

    for q in questions:
        result = await generator.generate(q.strip())
        print(f"Q: {q.strip()}")
        print(f"A: {result['answer'][:100]}...")
        print(f"Confidence: {result['confidence']:.2f}\n")

asyncio.run(test_all())
```

3. **Iteriere:**
   - Ändere Config
   - Lauf Test
   - Bewerte Ergebnisse
   - Repeat

### 3.2 Systematisches Vorgehen

**Woche 1: Retrieval**
- [ ] Embedding-Modell testen (3 verschiedene)
- [ ] Chunk-Größen experimentieren (256, 512, 768, 1024)
- [ ] Fusion-Gewichte tunen (0.5/0.5, 0.7/0.3, 0.3/0.7)

**Woche 2: Generation**
- [ ] Prompts A/B-Testen
- [ ] Temperatur-Settings
- [ ] Provider vergleichen (OpenAI vs Claude)

**Woche 3: End-to-End**
- [ ] 50+ Testfragen sammeln
- [ ] Baseline-Performance messen
- [ ] Optimale Config finalisieren

---

## 📊 Schritt 4: Messung & Evaluation

### Ground Truth erstellen

```python
# Create test_cases.json
[
  {
    "question": "Wie viele ECTS hat Algorithmen?",
    "expected_answer": "9 ECTS",
    "category": "ects_lookup"
  },
  {
    "question": "Welche Voraussetzungen für BA-Arbeit?",
    "expected_answer": "120 ECTS aus Pflichtmodulen",
    "category": "prerequisites"
  }
]
```

### Metriken tracken

```python
# evaluation.py
import json

results = {
    "config": "baseline",
    "avg_confidence": 0.75,
    "abstain_rate": 0.1,
    "correct_answers": 45,
    "total_questions": 50,
    "accuracy": 0.9
}

with open('results.json', 'w') as f:
    json.dump(results, f)
```

### A/B-Testing

```bash
# Config A: baseline
cp config/config.yaml config/config_baseline.yaml

# Config B: optimized
# ... ändere Settings ...
cp config/config.yaml config/config_optimized.yaml

# Teste beide
python debug.py test-suite --config config/config_baseline.yaml > results_a.txt
python debug.py test-suite --config config/config_optimized.yaml > results_b.txt

# Vergleiche
diff results_a.txt results_b.txt
```

---

## 🚀 Quick Wins (Top 5)

### 1. **Chunk-Größe halbieren**
```yaml
chunking:
  semantic:
    max_chunk_size: 256  # Runter von 512
```
→ Oft 20-30% bessere Precision

### 2. **BM25-Gewicht erhöhen**
```yaml
retrieval:
  hybrid:
    sparse_weight: 0.5  # Hoch von 0.3
```
→ Besser bei exakten Keywords

### 3. **Temperatur auf 0**
```yaml
llm:
  openai:
    temperature: 0.0
```
→ Maximale Faktentreue

### 4. **Mehr Kontext**
```yaml
retrieval:
  final_top_k: 7  # Hoch von 5
```
→ LLM hat mehr Informationen

### 5. **System-Prompt verschärfen**
→ Siehe Fix C1

---

## 🆘 Wenn gar nichts hilft

### Letzte Maßnahmen:

1. **Dokumente prüfen:**
```bash
python debug.py analyze-chunks
```
Sind überhaupt relevante Infos drin?

2. **Neu scrapen:**
```bash
python pipeline.py clear-cache
python pipeline.py build-index
```

3. **LLM direkt testen:**
```python
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "Wie viele ECTS hat Algorithmen? Info: Das Modul umfasst 9 ECTS."}
    ]
)
print(response.choices[0].message.content)
```
Funktioniert LLM überhaupt?

4. **Community fragen:**
- GitHub Issues
- Discord: LangChain / RAG Community
- Reddit: r/MachineLearning

---

## 📈 Erfolgs-Metriken

### Gutes RAG-System:
- ✅ Confidence >0.8 bei klaren Fragen
- ✅ Abstaining bei Out-of-Scope (>95%)
- ✅ Richtige Quellenangaben (100%)
- ✅ Keine Halluzinationen bei Zahlen (>98%)
- ✅ Retrieval <500ms
- ✅ Generation <3s

### Messe regelmäßig:
```bash
python debug.py test-suite  # Wöchentlich
```

Logge Ergebnisse und tracke Verbesserungen!

---

**Pro-Tip:** Starte mit **einer** Frage, die gut funktionieren sollte, und optimiere bis diese perfekt ist. Dann generalisiere!

Viel Erfolg! 🚀
