import sys

for line in sys.stdin:
  line = line.strip('\n').split(' ')
  if len(line) == 1:
    print('0')
  else:
    labels = [f'{float(i)/(len(line)-1):.4f}' for i in range(len(line))]
    print(' '.join(labels))
