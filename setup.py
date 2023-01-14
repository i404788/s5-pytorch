from setuptools import setup, find_packages

setup(
  name = 's5-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.1.0',
  license='MIT',
  description = 'S5 - Simplified State Space Layers for Sequence Modeling - Pytorch',
  author = 'Ferris Kwaijtaal',
  author_email = 'ferris+gh@devdroplets.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/audiolm-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'audio generation'
  ],
  install_requires=[
    'einops>=0.6',
    'scipy',
    'torch>=1.13',
    'jax'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.10',
  ],
)

