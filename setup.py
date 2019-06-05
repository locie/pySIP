from setuptools import setup, find_packages


with open('README.md', 'r') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as f:
    requirements = f.read().split()

setup(
    name='bopt',
    version='0.1',
    description='Thermal bayesian optimization',
    url='https://gitlab.lancey.fr/ai/bayesian-optimization',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Operating System :: OS Independent',
        'Development Status :: 4 - Beta'
    ]
)
