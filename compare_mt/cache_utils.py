from functools import lru_cache
from nltk.stem.porter import PorterStemmer

def extract_cache_dicts(cache_dicts, key_list, num_out):
  if cache_dicts is not None:
    if len(cache_dicts) != num_out:
       raise ValueError(f'Length of cache_dicts should be equal to the number of output files!')
    if len(key_list) == 1:
      return [c[key_list[0]] for c in cache_dicts]
    return zip(*[[c[k] for k in key_list] for c in cache_dicts]) 

  return [None]*len(key_list)

def return_cache_dict(key_list, value_list):
  for v in value_list:
    if len(v) != 1:
      raise ValueError(f'Only support caching for one system at a time!')
  cache_dict = {k:v[0] for (k, v) in zip(key_list, value_list)}
  return cache_dict

class CachedPorterStemmer(PorterStemmer):
  """A wrapper class for PorterStemmer that uses LRU cache to reduce latency"""
  def __init__(self, mode=PorterStemmer.NLTK_EXTENSIONS):
    super().__init__(mode)

  @lru_cache(maxsize=50000)
  def stem(self, word, to_lowercase=True):
    return super().stem(word, to_lowercase)