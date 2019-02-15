from collections import defaultdict
import sys

cnts = defaultdict(lambda: 0)
for line in sys.stdin:
  for word in line.strip().split():
    cnts[word] += 1

for k, v in sorted(cnts.items(), key=lambda x: -x[1]):
  print(f'{k}\t{v}')
