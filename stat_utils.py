
def extract_salient_features(dict1, dict2, alpha=1.0):
  """
  Score salient features given to dictionaries.

  Args:
    dict1: First set of feature coutns
    dict2: Second set of feature counts
    alpha: The amount of smoothing (default 1 to Laplace smoothed probabilities)

  Returns:
    Laplace smoothed differences between features
  """
  all_keys = set(dict1.keys()) | set(dict2.keys())
  scores = {}
  for k in all_keys:
    scores[k] = (dict1[k]+alpha) / (dict1[k] + dict2[k] + 2*alpha)
  return scores