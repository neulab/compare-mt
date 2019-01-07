
def iterate_tokens(filename):
  with open(filename, "r") as f:
    for line in f:
      yield line.strip().split()

def load_tokens(filename):
  return list(iterate_tokens(filename))
