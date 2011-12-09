from setuptools import setup, find_packages
import sys, os

version = '0.0'

setup(name='CljPy',
      version=version,
      description="Clojure functions in Python",
      long_description="""\
Ever started work on a Python project but missed the functional tools that come with Python?  Just import cljpy and the functions you love will be at your fingertips again!""",
      classifiers=[], # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
      keywords='clojure python functional',
      author='Jim Blomo',
      author_email='jim.blomo@gmail.com',
      url='https://github.com/jblomo/CljPy',
      license='EPL',
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      include_package_data=True,
      zip_safe=True,
      install_requires=[
          # -*- Extra requirements: -*-
      ],
      entry_points="""
      # -*- Entry points: -*-
      """,
      )
