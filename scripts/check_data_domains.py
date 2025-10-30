#!/usr/bin/env python3
import json
from collections import Counter

data = json.load(open("/home/rocm-user/AMD_Hackathon/data/curriculum/train_45k.json"))
domains = Counter([q.get("domain", "unknown") for q in data])

print(f"Total questions: {len(data)}")
print("\nDomain distribution:")
for domain, count in sorted(domains.items(), key=lambda x: -x[1]):
    print(f"  {domain}: {count}")
