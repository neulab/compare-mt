def parse_profile(profile):
  kargs = {}
  for kv in profile.split(','):
    k, v = kv.split('=')
    kargs[k] = v
  return kargs

def parse_compare_directions(compare_directions):
  direcs = []
  for direc in compare_directions.split(';'):
    left, right = direc.split('-')
    left, right = int(left), int(right)
    direcs.append((left, right))
  return direcs

