from setuptools import setup

setup(name='vaevictis',
      version='0.2.1',
      description='test',
      install_requires=["annoy","numba",
      "tensorflow","tqdm","scipy","multiprocessing",
      "collections","operator","time","numpy",
      "random","sys", "os", "json"],
      packages=['vaevictis'],
      zip_safe=False)
