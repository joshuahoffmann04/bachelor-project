"""Pytest configuration and fixtures for RAG system tests."""

import pytest
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Provide a sample configuration for testing.

    Returns:
        Sample configuration dictionary
    """
    return {
        'chunking': {
            'strategy': 'hybrid',
            'semantic': {
                'enabled': True,
                'min_chunk_size': 100,
                'max_chunk_size': 512
            },
            'sliding_window': {
                'chunk_size': 512,
                'overlap': 102
            }
        },
        'embeddings': {
            'model': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
            'batch_size': 32,
            'device': 'cpu'
        },
        'retrieval': {
            'final_top_k': 7,
            'hybrid': {
                'enabled': True,
                'dense_weight': 0.8,
                'sparse_weight': 0.2
            }
        },
        'llm': {
            'provider': 'openai',
            'openai': {
                'model': 'gpt-4-turbo-preview',
                'temperature': 0.1,
                'max_tokens': 1000
            }
        },
        'prompts': {
            'abstaining': {
                'threshold': 0.75
            }
        }
    }


@pytest.fixture
def sample_text() -> str:
    """Provide sample text for testing.

    Returns:
        Sample German text about study regulations
    """
    return """Das Modul Algorithmen und Datenstrukturen ist ein Pflichtmodul im Bachelorstudiengang Informatik.

Es umfasst 8 ECTS-Punkte und wird im zweiten Semester angeboten.

Voraussetzungen:
- Grundlagen der Programmierung
- Mathematik 1

Die Prüfungsleistung besteht aus einer schriftlichen Klausur von 120 Minuten Dauer.

Modulverantwortlich: Prof. Dr. Müller"""


@pytest.fixture
def sample_text_long() -> str:
    """Provide long sample text for chunking tests.

    Returns:
        Long sample text
    """
    paragraphs = []
    for i in range(20):
        paragraphs.append(
            f"Absatz {i+1}: Dies ist ein Beispieltext für Tests. "
            f"Er enthält mehrere Sätze und ist lang genug, um beim Chunking "
            f"in mehrere Teile aufgeteilt zu werden. "
            f"Jeder Absatz hat etwa die gleiche Länge und Struktur."
        )
    return "\n\n".join(paragraphs)


@pytest.fixture
def sample_document():
    """Provide a sample Document object for testing.

    Returns:
        Sample Document
    """
    from src.preprocessing.document import Document, SourceType

    return Document(
        doc_id="test_doc_001",
        title="Test Prüfungsordnung",
        content="Das Modul hat 6 ECTS.\n\nVoraussetzung ist Mathematik 1.",
        source_type=SourceType.PDF,
        source_path="test.pdf"
    )


@pytest.fixture
def sample_chunks():
    """Provide sample DocumentChunk objects for testing.

    Returns:
        List of sample chunks
    """
    from src.preprocessing.document import DocumentChunk, SourceType, ContentType

    return [
        DocumentChunk(
            chunk_id="test_doc_001_c0",
            content="Das Modul Algorithmen und Datenstrukturen hat 8 ECTS.",
            doc_id="test_doc_001",
            source_type=SourceType.PDF,
            content_type=ContentType.PARAGRAPH,
            page=1,
            section="Module",
            source_doc="Modulhandbuch",
            char_count=55,
            tokens=15
        ),
        DocumentChunk(
            chunk_id="test_doc_001_c1",
            content="Voraussetzung ist Grundlagen der Programmierung.",
            doc_id="test_doc_001",
            source_type=SourceType.PDF,
            content_type=ContentType.PARAGRAPH,
            page=1,
            section="Voraussetzungen",
            source_doc="Modulhandbuch",
            char_count=49,
            tokens=12
        ),
        DocumentChunk(
            chunk_id="test_doc_001_c2",
            content="Die Prüfung ist eine schriftliche Klausur von 120 Minuten.",
            doc_id="test_doc_001",
            source_type=SourceType.PDF,
            content_type=ContentType.PARAGRAPH,
            page=2,
            section="Prüfung",
            source_doc="Modulhandbuch",
            char_count=59,
            tokens=14
        )
    ]


@pytest.fixture
def sample_query() -> str:
    """Provide a sample query for testing.

    Returns:
        Sample query string
    """
    return "Wie viele ECTS hat das Modul Algorithmen und Datenstrukturen?"


@pytest.fixture
def temp_config_file(tmp_path, sample_config):
    """Create a temporary config file for testing.

    Args:
        tmp_path: pytest tmp_path fixture
        sample_config: Sample configuration

    Returns:
        Path to temporary config file
    """
    import yaml

    config_file = tmp_path / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(sample_config, f)

    return config_file


@pytest.fixture
def temp_test_dataset(tmp_path):
    """Create a temporary test dataset for testing.

    Args:
        tmp_path: pytest tmp_path fixture

    Returns:
        Path to temporary test dataset file
    """
    import json

    test_cases = [
        {
            "question": "Wie viele ECTS hat Algorithmen?",
            "ground_truth_answer": "8 ECTS",
            "ects_value": 8,
            "category": "ects_lookup",
            "difficulty": "easy"
        },
        {
            "question": "Was sind die Voraussetzungen?",
            "category": "prerequisites",
            "difficulty": "medium"
        }
    ]

    dataset_file = tmp_path / "test_set.jsonl"
    with open(dataset_file, 'w') as f:
        for tc in test_cases:
            f.write(json.dumps(tc) + '\n')

    return dataset_file


# Markers for test categories
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "requires_api: mark test as requiring API keys")
    config.addinivalue_line("markers", "requires_indices: mark test as requiring built indices")
