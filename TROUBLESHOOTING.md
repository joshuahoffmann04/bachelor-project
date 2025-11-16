# 🆘 Troubleshooting Checkliste

Schnelle Lösungen für häufige Probleme.

## ⚡ Quick Fixes (Probiere diese zuerst!)

### 1. "Indices not found"
```bash
python pipeline.py build-index
```

### 2. "Schlechte Ergebnisse"
```bash
# Diagnose
python debug.py full-report

# Quick-Fix
python optimize.py interactive
```

### 3. "Zu langsam"
```bash
python optimize.py slow_performance
```

### 4. "Halluzinationen"
```bash
python optimize.py hallucinations
```

### 5. "Sagt immer 'Ich weiß es nicht'"
```bash
python optimize.py abstains_too_much
```

---

## 🔍 Detaillierte Diagnose

### Schritt 1: System-Check

```bash
# Check 1: Sind Indices vorhanden?
ls -lh data/processed/

# Sollte zeigen:
# - faiss_index.bin
# - chunks.pkl
# - bm25_index.pkl

# Check 2: Sind Dokumente vorhanden?
ls -lh data/raw/

# Sollte PDF(s) zeigen

# Check 3: Wurde gescraped?
ls -lh data/scraped/

# Sollte .json Dateien zeigen
```

### Schritt 2: Chunk-Analyse

```bash
python debug.py analyze-chunks
```

**Erwartete Werte:**
- Total Chunks: >50 (je nach Dokumentgröße)
- Avg Tokens/Chunk: 200-512
- Content Types: Mix aus paragraph, table, list

**❌ Problem:** Nur 5 Chunks
- **Fix:** Dokument ist zu klein oder Chunking fehlgeschlagen
- **Action:** `python pipeline.py build-index` neu laufen lassen

**❌ Problem:** Avg Tokens >1000
- **Fix:** Chunks zu groß
- **Action:** `python optimize.py chunks_too_large`

### Schritt 3: Retrieval-Test

```bash
python debug.py test-retrieval "Wie viele ECTS hat Algorithmen?"
```

**Gutes Ergebnis:**
- Top-3 Scores: >0.5
- Chunks enthalten relevante Info
- Hybrid besser als Dense/Sparse allein

**❌ Problem:** Alle Scores <0.3
- **Fix:** Embedding-Modell passt nicht
- **Action:**
  ```yaml
  embeddings:
    model: deutsche-telekom/gbert-large-paraphrase-cosine
  ```
  Dann: `python pipeline.py build-index`

**❌ Problem:** BM25 viel besser als Dense
- **Fix:** Gewichtung anpassen
- **Action:** `python optimize.py retrieval_poor`

---

## ✅ Success Checklist

Dein System läuft gut wenn:

- [ ] `python debug.py analyze-chunks` zeigt >50 Chunks
- [ ] `python debug.py test-retrieval "Test"` findet relevante Chunks
- [ ] `python debug.py test-suite` zeigt Avg Confidence >0.6
- [ ] Antworten sind faktisch korrekt (manuell geprüft)
- [ ] Quellenangaben sind präzise
- [ ] Out-of-Scope Fragen werden abstained
- [ ] Retrieval <500ms, Generation <3s

Wenn ALLE gecheckt: **🎉 Glückwunsch! Dein RAG-System läuft!**

---

**Letzter Tipp:** Bei komplexen Problemen → Lies OPTIMIZATION_GUIDE.md für systematisches Tuning!
