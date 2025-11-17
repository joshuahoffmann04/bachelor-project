"""Unit tests for evaluation metrics."""

import pytest
from src.evaluation.metrics import CustomMetrics, _extract_ects


@pytest.mark.unit
class TestECTSAccuracy:
    """Tests for ECTS accuracy metric."""

    def test_exact_match(self):
        """Test exact ECTS value match."""
        answer = "Das Modul hat 6 ECTS."
        ground_truth = "6 ECTS"

        result = CustomMetrics.ects_accuracy(answer, ground_truth, tolerance=0)

        assert result.score == 1.0
        assert result.details['answer_ects'] == 6.0
        assert result.details['truth_ects'] == 6.0

    def test_mismatch(self):
        """Test ECTS value mismatch."""
        answer = "Das Modul hat 8 ECTS."
        ground_truth = "6 ECTS"

        result = CustomMetrics.ects_accuracy(answer, ground_truth, tolerance=0)

        assert result.score == 0.0
        assert result.details['difference'] == 2.0

    def test_with_tolerance(self):
        """Test ECTS matching with tolerance."""
        answer = "Das Modul hat 6 ECTS."
        ground_truth = "5 ECTS"

        # Should match with tolerance of 1
        result = CustomMetrics.ects_accuracy(answer, ground_truth, tolerance=1.0)
        assert result.score == 1.0

        # Should not match with tolerance of 0
        result = CustomMetrics.ects_accuracy(answer, ground_truth, tolerance=0.0)
        assert result.score == 0.0

    def test_no_ects_in_answer(self):
        """Test when no ECTS mentioned in answer."""
        answer = "Das Modul ist ein Pflichtmodul."
        ground_truth = "6 ECTS"

        result = CustomMetrics.ects_accuracy(answer, ground_truth)

        assert result.score == 0.0
        assert result.details['reason'] == 'missing_ects'

    def test_alternative_formats(self):
        """Test various ECTS formats."""
        test_cases = [
            ("6 ECTS", "6 ECTS", True),
            ("6 LP", "6 ECTS", True),  # Leistungspunkte
            ("6 Credits", "6 ECTS", True),
            ("ECTS: 6", "6 ECTS", True),
        ]

        for answer, truth, should_match in test_cases:
            result = CustomMetrics.ects_accuracy(answer, truth, tolerance=0)
            assert result.score == (1.0 if should_match else 0.0), f"Failed for: {answer}"


@pytest.mark.unit
class TestReferenceQuality:
    """Tests for reference quality metric."""

    def test_with_citations(self):
        """Test answer with proper citations."""
        answer = "Laut Quelle [1] hat das Modul 6 ECTS. Siehe auch [2]."
        contexts = ["Context 1", "Context 2"]

        result = CustomMetrics.reference_quality(answer, contexts, required_citations=1)

        assert result.score == 1.0
        assert result.details['num_citations'] == 2

    def test_no_citations(self):
        """Test answer without citations."""
        answer = "Das Modul hat 6 ECTS."
        contexts = ["Context 1"]

        result = CustomMetrics.reference_quality(answer, contexts, required_citations=1)

        assert result.score == 0.0
        assert result.details['reason'] == 'no_citations'

    def test_invalid_citation_numbers(self):
        """Test answer with invalid citation numbers."""
        answer = "Siehe Quelle [5]."  # Only 2 contexts available
        contexts = ["Context 1", "Context 2"]

        result = CustomMetrics.reference_quality(answer, contexts, required_citations=1)

        assert result.score == 0.5  # Partial credit
        assert 5 in result.details['invalid_citations']

    def test_required_citations(self):
        """Test required citations threshold."""
        answer = "Laut [1] hat das Modul 6 ECTS."
        contexts = ["Context 1", "Context 2"]

        # Should pass with 1 required
        result = CustomMetrics.reference_quality(answer, contexts, required_citations=1)
        assert result.score == 1.0

        # Should fail with 3 required
        result = CustomMetrics.reference_quality(answer, contexts, required_citations=3)
        assert result.score < 1.0


@pytest.mark.unit
class TestAbstainingRate:
    """Tests for abstaining rate metric."""

    def test_no_abstaining(self):
        """Test when no answers abstain."""
        answers = [
            "Das Modul hat 6 ECTS.",
            "Die Prüfung ist schriftlich.",
            "Voraussetzung ist Mathematik 1."
        ]

        result = CustomMetrics.abstaining_rate(answers)

        assert result.score == 0.0
        assert result.details['abstaining_count'] == 0

    def test_all_abstaining(self):
        """Test when all answers abstain."""
        answers = [
            "Ich weiß nicht.",
            "Kann ich nicht beantworten.",
            "Keine Information verfügbar."
        ]

        result = CustomMetrics.abstaining_rate(answers)

        assert result.score == 1.0
        assert result.details['abstaining_count'] == 3

    def test_partial_abstaining(self):
        """Test with mixed answers."""
        answers = [
            "Das Modul hat 6 ECTS.",
            "Ich weiß nicht.",
            "Die Prüfung ist schriftlich."
        ]

        result = CustomMetrics.abstaining_rate(answers)

        assert 0.3 < result.score < 0.4  # 1 out of 3
        assert result.details['abstaining_count'] == 1

    def test_custom_keywords(self):
        """Test with custom abstaining keywords."""
        answers = ["Sorry, I don't know."]
        keywords = ["sorry", "don't know"]

        result = CustomMetrics.abstaining_rate(answers, abstaining_keywords=keywords)

        assert result.score == 1.0


@pytest.mark.unit
class TestHallucinationDetection:
    """Tests for hallucination detection metric."""

    def test_no_hallucination(self):
        """Test answer grounded in context."""
        answer = "Das Modul hat 6 ECTS."
        contexts = ["Das Modul umfasst 6 ECTS-Punkte."]

        result = CustomMetrics.hallucination_detection(answer, contexts)

        assert result.score == 0.0  # No hallucination
        assert not result.details['hallucinated_numbers']

    def test_hallucinated_numbers(self):
        """Test answer with numbers not in context."""
        answer = "Das Modul hat 12 ECTS."
        contexts = ["Das Modul umfasst 6 ECTS-Punkte."]

        result = CustomMetrics.hallucination_detection(answer, contexts)

        assert result.score > 0.0  # Hallucination detected
        assert '12' in result.details['hallucinated_numbers']

    def test_hallucinated_ects(self):
        """Test specifically ECTS hallucination."""
        answer = "Das Modul hat 8 ECTS."
        contexts = ["Das Modul hat 6 ECTS."]

        result = CustomMetrics.hallucination_detection(answer, contexts)

        assert result.score > 0.0
        assert result.details['ects_hallucinated'] is True
        assert result.details['answer_ects'] == 8.0
        assert 6.0 in result.details['context_ects']

    def test_multiple_contexts(self):
        """Test with multiple context chunks."""
        answer = "Das Modul hat 6 ECTS und dauert 120 Minuten."
        contexts = [
            "Das Modul umfasst 6 ECTS.",
            "Die Prüfung dauert 120 Minuten."
        ]

        result = CustomMetrics.hallucination_detection(answer, contexts)

        assert result.score == 0.0  # All facts are in contexts


@pytest.mark.unit
class TestExtractECTS:
    """Tests for ECTS extraction helper function."""

    def test_standard_format(self):
        """Test standard ECTS format."""
        assert _extract_ects("Das Modul hat 6 ECTS.") == 6.0
        assert _extract_ects("8 ECTS") == 8.0

    def test_leistungspunkte(self):
        """Test LP (Leistungspunkte) format."""
        assert _extract_ects("Das Modul hat 6 LP.") == 6.0
        assert _extract_ects("8 Leistungspunkte") == 8.0

    def test_credits(self):
        """Test Credits format."""
        assert _extract_ects("6 Credits") == 6.0
        assert _extract_ects("Credits: 8") == 8.0

    def test_colon_format(self):
        """Test colon-separated format."""
        assert _extract_ects("ECTS: 6") == 6.0
        assert _extract_ects("LP: 8") == 8.0

    def test_decimal_values(self):
        """Test decimal ECTS values."""
        assert _extract_ects("4.5 ECTS") == 4.5
        assert _extract_ects("7.5 LP") == 7.5

    def test_no_ects(self):
        """Test text without ECTS."""
        assert _extract_ects("Das Modul ist wichtig.") is None
        assert _extract_ects("Keine Angabe") is None

    def test_case_insensitive(self):
        """Test case insensitive matching."""
        assert _extract_ects("6 ects") == 6.0
        assert _extract_ects("6 Ects") == 6.0
        assert _extract_ects("6 lp") == 6.0
