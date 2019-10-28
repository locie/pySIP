from setuptools import setup, find_packages
from pysip import __version__

with open('README.md', 'r') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as f:
    requirements = f.read().split()

setup(
    name='pysip',
    version=__version__,
    description='PySIP',
    url='https://gitlab.lancey.fr/ai/models/pysip',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Operating System :: OS Independent',
        'Development Status :: 4 - Beta',
    ],
)
