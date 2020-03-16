from setuptools import setup

setup(name='vaevictis',
      version='0.2.1',
      description='test',
      install_requires=["annoy","numba","tensorflow","tqdm"],
      packages=['vaevictis'],
      zip_safe=False)
