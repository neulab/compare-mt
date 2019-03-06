def parse_profile(profile):
  kargs = {}
  try:
    for kv in profile.split(','):
      k, v = kv.split('=')
      kargs[k] = v
  except ValueError:
    # more informative error message
    raise ValueError(
      f"Failed to parse profile: {profile}. The expected format is:"
      " \"key1=value1,key2=value2,[...]\""
    )
  return kargs

def parse_compare_directions(compare_directions):
  direcs = []
  try:
    for direc in compare_directions.split(';'):
      left, right = direc.split('-')
      left, right = int(left), int(right)
      direcs.append((left, right))
  except ValueError:
    # more informative error message
    raise ValueError(
      f"Failed to parse directions: {compare_directions}."
      " The expected format is: \"left1-right1;left2-right2;[...]\""
    )
  return direcs

def parse_files(filenames):
  files = []
  for f in filenames.split(';'):
    files.append(f)
  return files

def parse_intfloat(s):
  try:
    return int(s)
  except ValueError:
    return float(s)