# This script is a simple script to interleave the lines of multiple systems
# It can be used like
#  python interleave.py ref.txt sys1.txt sys2.txt

import sys
import itertools

filenames = sys.argv[1:]
files = [open(x, 'r') for x in filenames]
assert all(files), f'Could not open all files in {filenames}'

for lines in itertools.zip_longest(*files):
  for line in lines:
    print(line.strip('\n'))
  print()
