from setuptools import setup, find_packages
import unittest
import codecs
import re


def test_suite():
  test_loader = unittest.TestLoader()
  test_suite = test_loader.discover("compare_mt/tests", pattern="test_*.py")

  return test_suite


def find_version(*file_paths):
  """Find version in compare_mt/__init__.py"""
  os.path.abspath(os.path.dirname(__file__))
  with codecs.open(os.path.join(here, *parts), 'r') as fp:
    version_file = fp.read()
    version_match = re.search(
      r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
  raise RuntimeError("Unable to find version string.")


setup(
  name="compare_mt",
  version="0.1",
  description="Holistic comparison of the output of text generation models",
  long_description=codecs.open("README.md", encoding="utf-8").read(),
  long_description_content_type="text/markdown",
  url="https://github.com/neulab/compare-mt",
  author="Graham Neubig",
  license="[TODO]",
  test_suite="setup.test_suite",
  classifiers=[
  "Intended Audience :: Developers",
  "Topic :: Text Processing",
  "Topic :: Scientific/Engineering :: Artificial Intelligence",
  "License :: OSI Approved :: [TODO]",
  "Programming Language :: Python :: 3",
  ],
  packages=find_packages(),
  entry_points={
    "console_scripts": [
      "compare-mt=compare_mt.compare_mt:main",
      "compare-ll=compare_mt.compare_ll:main",
    ],
  },
  include_package_data=True
)