from setuptools import setup, find_packages

setup(
    name='s5-pytorch',
    packages=find_packages(exclude=[]),
    version='0.2.0',
    license='MIT',
    description='S5 - Simplified State Space Layers for Sequence Modeling - Pytorch',
    author='Ferris Kwaijtaal',
    author_email='ferris+gh@devdroplets.com',
    long_description_content_type='text/markdown',
    long_description=open('README.md', 'r').read(),
    url='https://github.com/i404788/s5-pytorch',
    keywords=[
        'artificial intelligence',
        'deep learning',
        'transformers',
        'attention mechanism',
        'audio generation'
    ],
    install_requires=[
        'einops>=0.6',
        'scipy',
        'torch>=2',
    ],
    extra_requires={
      "dev": ["jax"], 
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
    ],
)
