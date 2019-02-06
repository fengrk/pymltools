# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

# version info
VERSION = (2019, 2, 6)
VERSION_STATUS = ""
VERSION_TEXT = ".".join(str(x) for x in VERSION) + VERSION_STATUS

setup(name="pymltools",
      version=VERSION_TEXT,
      description="machine learning tools for python3.6",
      long_description=open("README.md", "r").read(),
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Intended Audience :: Other Audience',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Natural Language :: English',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Topic :: Scientific/Engineering',
          'Topic :: Software Development :: Libraries',
          'Topic :: Utilities',
      ],
      install_requires=[
          "tensorflow",
          "pyxtools",
          "h5py",
          "numpy",
          "Pillow"
      ],
      author="frkhit",
      url="https://github.com/frkhit/pymltools",
      author_email="frkhit@gmail.com",
      license="MIT",
      packages=find_packages(),
      package_data={'': ["LICENSE", "README.md", "MANIFEST.in"]},
      include_package_data=True, )
