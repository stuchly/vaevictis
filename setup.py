from setuptools import setup

setup(name='vaevictis',
      version='0.2.1',
      description='test',
      install_requires=["annoy","numba",
      "tensorflow","tqdm","scipy",#"multiprocessing","collections","operator","time","random","sys","os","json",
      "numpy"],
      packages=['vaevictis'],
      zip_safe=False)
