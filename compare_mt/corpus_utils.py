def iterate_tokens(filename):
  with open(filename, "r", encoding="utf-8") as f:
    for line in f:
      yield line.strip().split(' ')

def load_tokens(filename):
  return list(iterate_tokens(filename))

def iterate_nums(filename):
  with open(filename, "r", encoding="utf-8") as f:
    for line in f:
      yield [float(i) for i in line.strip().split(' ')]

def load_nums(filename):
  return list(iterate_nums(filename))

def iterate_alignments(filename):
  with open(filename, "r", encoding="utf-8") as f:
    for line in f:
      try:
        yield [(int(src),int(trg)) for (src,trg) in [x.split('-') for x in line.strip().split(' ')]]
      except:
        raise ValueError(f'Poorly formed alignment line in {filename}:\n{line}')

def load_alignments(filename):
  return list(iterate_alignments(filename))

def lower(inp):
  return inp.lower() if type(inp) == str else [lower(x) for x in inp]

def list2str(l):
  string = ''
  for i, s in enumerate(l):
    string = string + ' ' + str(s) if i != 0 else string + str(s)
  return string
  
def write_tokens(filename, ls):
  with open(filename, 'w') as f:
    for i, l in enumerate(ls):
      string = list2str(l)
      string = '\n' + string if i != 0 else string
      f.write(string)
  return string
