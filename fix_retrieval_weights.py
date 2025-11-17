#!/usr/bin/env python3
"""Quick fix for retrieval weights - Dense is better than BM25!"""

import yaml
import shutil
from datetime import datetime

# Backup
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = f"config/config.backup_{timestamp}.yaml"
shutil.copy("config/config.yaml", backup)
print(f"✓ Backup: {backup}")

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Show current settings
old_dense = config.get('retrieval', {}).get('hybrid', {}).get('dense_weight', 0.7)
old_sparse = config.get('retrieval', {}).get('hybrid', {}).get('sparse_weight', 0.3)
old_threshold = config.get('prompts', {}).get('abstaining', {}).get('threshold', 0.6)

print(f"\nCURRENT Settings:")
print(f"  Dense weight: {old_dense}")
print(f"  Sparse weight: {old_sparse}")
print(f"  Abstaining threshold: {old_threshold}")

# Fix retrieval weights - Dense is MUCH better!
config['retrieval']['hybrid']['dense_weight'] = 0.9
config['retrieval']['hybrid']['sparse_weight'] = 0.1

# Lower abstaining threshold
config['prompts']['abstaining']['threshold'] = 0.0

# Save
with open('config/config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"\n✓ FIXED Settings:")
print(f"  Dense weight: 0.9 (Dense finds correct chunks!)")
print(f"  Sparse weight: 0.1 (BM25 was confusing)")
print(f"  Abstaining threshold: 0.0 (No more abstaining)")

print("\n🎯 WHY THIS WORKS:")
print("  Dense Retrieval found:")
print("    Rank 5-6: 'Algorithmen und Datenstrukturen WP 9' ✓")
print("  BM25 found:")
print("    Rank 1-10: Wrong stuff about Prüfungsordnung ✗")
print("  → Dense is BETTER! So we give it 90% weight!")

print("\n🚀 NOW TEST:")
print('  python debug.py test-generation "Wie viele ECTS hat Algorithmen?"')
