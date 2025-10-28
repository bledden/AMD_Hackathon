#!/usr/bin/env python3
import json
from collections import Counter

with open('/workspace/data/curriculum/train_65k_merged.json') as f:
    data = json.load(f)

cats = Counter([q.get('category', q.get('domain', 'unknown')) for q in data])

print("Top 20 Categories:")
for cat, count in sorted(cats.items(), key=lambda x: -x[1])[:20]:
    print(f"{cat}: {count}")
