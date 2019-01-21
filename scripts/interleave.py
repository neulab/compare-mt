import sys
import itertools


filenames = sys.argv[1:]
files = [open(x, 'r') for x in filenames]
assert all(files), f'Could not open all files in {filenames}'

for lines in itertools.zip_longest(*files):
  for line in lines:
    print(line.strip('\n'))
  print()
