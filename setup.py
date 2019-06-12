from setuptools import setup, find_packages
import unittest
import codecs
import re
import os


def test_suite():
  test_loader = unittest.TestLoader()
  test_suite = test_loader.discover("compare_mt/tests", pattern="test_*.py")

  return test_suite


def find_version():
  """Find version in compare_mt/__init__.py"""
  here = os.path.abspath(os.path.dirname(__file__))
  with codecs.open(os.path.join(here, "compare_mt", "__init__.py"), 'r') as fp:
    version_file = fp.read()
    version_match = re.search(
      r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
  raise RuntimeError("Unable to find version string.")


setup(
  name="compare_mt",
  version=find_version(),
  description="Holistic comparison of the output of text generation models",
  long_description=codecs.open("README.md", encoding="utf-8").read(),
  long_description_content_type="text/markdown",
  url="https://github.com/neulab/compare-mt",
  author="Graham Neubig",
  license="BSD 3-Clause",
  test_suite="setup.test_suite",
  classifiers=[
  "Intended Audience :: Developers",
  "Topic :: Text Processing",
  "Topic :: Scientific/Engineering :: Artificial Intelligence",
  "License :: OSI Approved :: BSD License",
  "Programming Language :: Python :: 3",
  ],
  packages=find_packages(),
  entry_points={
    "console_scripts": [
      "compare-mt=compare_mt.compare_mt_main:main",
      "compare-ll=compare_mt.compare_ll_main:main",
    ],
  },
  install_requires=[
    "nltk>=3.2",
    "numpy",
    "matplotlib",
    "absl-py",
    "sacrebleu"
  ],
  include_package_data=True,
)
