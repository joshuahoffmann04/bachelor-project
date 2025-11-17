# Test Datasets for RAG System Evaluation

This directory contains test datasets for evaluating the RAG (Retrieval Augmented Generation) system.

## Dataset Format

All test datasets use **JSONL** (JSON Lines) format, where each line is a valid JSON object representing one test case.

### Schema

Each test case should contain the following fields:

```json
{
  "question": "User's question in German",
  "ground_truth_answer": "Expected answer (optional)",
  "expected_chunks": ["List of expected document IDs or sections"],
  "category": "Test category",
  "difficulty": "easy|medium|hard",
  "ects_value": 6,
  "metadata": {
    "module": "Module name",
    "source": "Which document",
    "notes": "Additional notes"
  }
}
```

### Required Fields

- `question` (string): The question to ask the system
- `category` (string): One of: ects_lookup, prerequisites, deadlines, exam_formats, out_of_scope

### Optional Fields

- `ground_truth_answer` (string): The expected answer for comparison
- `expected_chunks` (array): Expected document chunks/sections to be retrieved
- `difficulty` (string): Difficulty level - easy, medium, or hard
- `ects_value` (number): Expected ECTS value (for ects_lookup category)
- `metadata` (object): Any additional metadata

## Test Categories

### 1. ECTS Lookups (`ects_lookups.jsonl`)

Tests for retrieving ECTS credit information for modules.

**Example:**
```json
{"question": "Wie viele ECTS hat das Modul Algorithmen und Datenstrukturen?", "ground_truth_answer": "8 ECTS", "ects_value": 8, "category": "ects_lookup", "difficulty": "easy"}
```

### 2. Prerequisites (`prerequisites.jsonl`)

Tests for module prerequisites and requirements.

**Example:**
```json
{"question": "Welche Voraussetzungen gibt es für das Modul Advanced Database Systems?", "category": "prerequisites", "difficulty": "medium"}
```

### 3. Deadlines (`deadlines.jsonl`)

Tests for examination deadlines, registration periods, etc.

**Example:**
```json
{"question": "Bis wann muss ich mich für die Bachelorarbeit anmelden?", "category": "deadlines", "difficulty": "medium"}
```

### 4. Exam Formats (`exam_formats.jsonl`)

Tests for examination types and formats.

**Example:**
```json
{"question": "Wie läuft die Prüfung im Modul Softwaretechnik ab?", "category": "exam_formats", "difficulty": "medium"}
```

### 5. Out of Scope (`out_of_scope.jsonl`)

Tests for questions that should NOT be answered (abstaining).

**Example:**
```json
{"question": "Was ist die beste Programmiersprache für Anfänger?", "ground_truth_answer": "System should abstain", "category": "out_of_scope", "difficulty": "easy"}
```

## Usage

### Loading Test Data

```python
import json
from pathlib import Path

def load_test_set(category: str) -> list:
    """Load test set by category."""
    file_path = Path(f"data/test_sets/{category}.jsonl")
    test_cases = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            test_cases.append(json.loads(line))

    return test_cases

# Load ECTS lookup tests
ects_tests = load_test_set("ects_lookups")
```

### Running Evaluation

```bash
# Evaluate single test set
python cli.py evaluate --test-set data/test_sets/ects_lookups.jsonl

# Evaluate all test sets
python cli.py evaluate --test-set data/test_sets/*.jsonl
```

## Adding New Test Cases

1. Choose the appropriate category file
2. Add a new JSON object on a new line
3. Ensure all required fields are present
4. Validate JSON syntax (use `jq` or similar tool)
5. Test with a small subset first

**Example workflow:**
```bash
# Add new test case
echo '{"question": "Neue Frage?", "category": "ects_lookup", "difficulty": "easy"}' >> data/test_sets/ects_lookups.jsonl

# Validate JSON
cat data/test_sets/ects_lookups.jsonl | jq empty

# Run evaluation
python cli.py evaluate --test-set data/test_sets/ects_lookups.jsonl
```

## Best Practices

1. **Diversity**: Include questions with varying:
   - Difficulty levels
   - Question phrasings
   - Expected answer lengths
   - Edge cases

2. **Balance**: Aim for balanced distribution:
   - Easy: 40%
   - Medium: 40%
   - Hard: 20%

3. **Ground Truth**: Provide ground truth answers when:
   - Testing specific fact retrieval (ECTS, dates)
   - Validating answer correctness
   - Measuring hallucination rates

4. **Out of Scope**: Include questions that:
   - Ask for opinions or recommendations
   - Request information not in documents
   - Are ambiguous or underspecified

5. **Version Control**:
   - Commit test sets to git
   - Document changes in commit messages
   - Tag major test set versions

## Statistics

Current test set sizes:
- ECTS Lookups: 10 questions
- Prerequisites: 10 questions
- Deadlines: 10 questions
- Exam Formats: 10 questions
- Out of Scope: 10 questions

**Total: 50 test questions**

Target: 100+ questions across all categories

## Maintenance

- Review and update test sets quarterly
- Remove outdated questions (e.g., old deadlines)
- Add new questions based on:
  - User feedback
  - Common queries
  - System failures
  - New features

## Contact

For questions or suggestions about test datasets, please open an issue on GitHub.
