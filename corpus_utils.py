
def iterate_tokens(filename):
  with open(filename, "r") as f:
    for line in f:
      yield line.strip().split()

def iterate_nums(filename):
  with open(filename, "r") as f:
    for line in f:
      yield [float(i) for i in line.strip().split()]

def load_tokens(filename):
  return list(iterate_tokens(filename))

def load_nums(filename):
  return list(iterate_nums(filename))

def lower(inp):
  return inp.lower() if type(inp) == str else [lower(x) for x in inp]
