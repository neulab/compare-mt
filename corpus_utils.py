
def iterate_tokens(filename):
  with open(filename, "r") as f:
    for line in f:
      yield line.strip().split()

def load_tokens(filename):
  return list(iterate_tokens(filename))

def iterate_scores(filename):
  with open(filename, "r") as f:
    for line in f:
      yield [float(score) for score in line.strip().split()]

def load_scores(filename):
  return list(iterate_scores(filename))
