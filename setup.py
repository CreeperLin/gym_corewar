from setuptools import setup, find_packages

setup(name='corewar_env',
      version='0.0.1',
      install_requires=['gym','Corewar'],
      packages=find_packages(),
)