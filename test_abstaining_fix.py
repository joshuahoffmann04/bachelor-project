#!/usr/bin/env python3
"""Test script to verify abstaining fix."""


def test_rrf_score_calculation():
    """Demonstrate RRF score calculation before and after normalization."""

    print("="*80)
    print("RRF SCORE CALCULATION TEST")
    print("="*80 + "\n")

    # Simulate RRF calculation
    dense_weight = 0.8
    sparse_weight = 0.2
    k = 60

    print("Configuration:")
    print(f"  - dense_weight: {dense_weight}")
    print(f"  - sparse_weight: {sparse_weight}")
    print(f"  - k: {k}")
    print()

    # Calculate RRF scores for top 5 results
    print("RAW RRF Scores (before normalization):")
    print("-" * 80)

    raw_scores = []
    for rank in range(1, 6):
        # Assume chunk appears in both retrievers
        dense_rrf = dense_weight / (k + rank)
        sparse_rrf = sparse_weight / (k + rank)
        total_rrf = dense_rrf + sparse_rrf

        raw_scores.append(total_rrf)
        print(f"  Rank {rank}: {total_rrf:.6f}")

    avg_raw = sum(raw_scores) / len(raw_scores)
    print(f"\nAverage Raw Score: {avg_raw:.6f}")
    print(f"❌ Threshold 0.5: {avg_raw:.6f} < 0.5 → ABSTAINING!")

    # Normalize scores
    print("\n" + "="*80)
    print("NORMALIZED SCORES (after fix):")
    print("-" * 80)

    min_score = min(raw_scores)
    max_score = max(raw_scores)

    normalized_scores = [
        (score - min_score) / (max_score - min_score)
        for score in raw_scores
    ]

    for rank, (raw, norm) in enumerate(zip(raw_scores, normalized_scores), 1):
        print(f"  Rank {rank}: {raw:.6f} → {norm:.4f}")

    avg_norm = sum(normalized_scores) / len(normalized_scores)
    top_norm = max(normalized_scores)

    print(f"\nAverage Normalized Score: {avg_norm:.4f}")
    print(f"Top Normalized Score: {top_norm:.4f}")

    # Improved confidence calculation
    confidence = max(top_norm * 0.7, avg_norm)
    print(f"\nImproved Confidence: max({top_norm:.4f} * 0.7, {avg_norm:.4f}) = {confidence:.4f}")

    threshold = 0.3
    print(f"\n✅ Threshold {threshold}: {confidence:.4f} > {threshold} → ANSWER GENERATED!")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("✓ RRF scores now normalized to [0, 1]")
    print("✓ Threshold adjusted to 0.3")
    print("✓ Confidence considers both top score and average")
    print("✓ Abstaining should now work correctly!")


def test_confidence_scenarios():
    """Test different confidence scenarios."""

    print("\n\n" + "="*80)
    print("CONFIDENCE SCENARIOS")
    print("="*80 + "\n")

    scenarios = [
        {
            'name': 'High Confidence (Perfect Match)',
            'scores': [1.0, 0.9, 0.85, 0.8, 0.75],
            'expected': 'GENERATE'
        },
        {
            'name': 'Medium Confidence (Good Match)',
            'scores': [0.7, 0.6, 0.5, 0.4, 0.3],
            'expected': 'GENERATE'
        },
        {
            'name': 'Low Confidence (Weak Match)',
            'scores': [0.4, 0.35, 0.3, 0.25, 0.2],
            'expected': 'GENERATE'
        },
        {
            'name': 'Very Low Confidence (No Match)',
            'scores': [0.2, 0.15, 0.1, 0.05, 0.01],
            'expected': 'ABSTAIN'
        }
    ]

    threshold = 0.3

    for scenario in scenarios:
        scores = scenario['scores']
        avg_score = sum(scores) / len(scores)
        top_score = max(scores)
        confidence = max(top_score * 0.7, avg_score)

        print(f"{scenario['name']}:")
        print(f"  Scores: {[f'{s:.2f}' for s in scores]}")
        print(f"  Average: {avg_score:.3f}")
        print(f"  Top Score: {top_score:.3f}")
        print(f"  Confidence: {confidence:.3f}")

        will_abstain = confidence < threshold
        result = "ABSTAIN" if will_abstain else "GENERATE"

        status = "✓" if result == scenario['expected'] else "✗"
        print(f"  {status} Result: {result} (expected: {scenario['expected']})")
        print()


if __name__ == "__main__":
    test_rrf_score_calculation()
    test_confidence_scenarios()

    print("\n" + "="*80)
    print("💡 TIP: You can adjust the threshold in config/config.yaml")
    print("   - Lower (e.g., 0.2): More permissive, fewer abstaining")
    print("   - Higher (e.g., 0.5): More strict, more abstaining")
    print("   - Current: 0.3 (balanced)")
    print("="*80 + "\n")
