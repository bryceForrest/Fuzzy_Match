from setuptools import setup

setup(
   name='Fuzzy_Match',
   version='1.0',
   description='Library for finding near matches for strings',
   packages=['Fuzzy_Match'],  #same as name
   install_requires=[
      'numpy==1.26.2',
      'pandas==2.1.4',
      'scikit-learn==1.3.2',
      'scipy==1.11.4',
   ], #external packages as dependencies
   author='Bryce Forrest',
   author_email='BForrest@lnw.com'
)