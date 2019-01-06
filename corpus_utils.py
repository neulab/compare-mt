
def iterate_tokens(filename, case_insensitive=False):
  with open(filename, "r") as f:
    for line in f:
      if not case_insensitive:
        yield line.strip().split()
      else:
        yield [x.lower() for x in line.strip().split()]

def load_tokens(filename, case_insensitive=False):
  return list(iterate_tokens(filename, case_insensitive))
