# Contribution guide

:arrow\_forward: **Requirements** [Python3.6](https://www.python.org/downloads/release/python-360/) • [AWS credentials](https://docs.aws.amazon.com/cli/latest/userguide/cli-config-files.html) • [Git](https://git-scm.com/)  • [DVC](https://dvc.org/)

``` bash
git clone git@gitlab.lancey.fr:ai/bayesian-optimization.git
cd bayesian-optimization
pip install -r requirements.txt
```

## :book: Build the documentation

``` bash
make -C doc html
```
Output files are located in `_build/html`. Continuous integration will automatically deployed a static version at this [adress](https://www.youtube.com/watch?v=dQw4w9WgXcQ)

## :robot: Test your code

``` bash
# Static analysis
invoke codestyle
invoke lint

# Unit tests
invoke tests core
invoke tests statespace
invoke tests ...

# Integration tests

# Data tests
invoke tests data --dvc
```

Before committing, you may ensure that all test status are green. In any case, commiting will trigger an [automatic test suite execution](https://gitlab.lancey.fr/ai/bayesian-optimization/pipelines). Merging into `master` and deploying releases should be done after validation. Adding or removing features from the codebase should be accompanied with an appropriate test suite update.


## :truck: Data sources

[DVC](https://dvc.org/) is the way to go to add large files to the repository without burden it. Data sources are copied to a [S3](https://aws.amazon.com/s3/) cloud storage solution whereas only  file hashes will be commited into `dvc` files. The commands are similar to Git, as shown in the [documentation](https://dvc.org/doc). The `pull` and `push` commands need valid AWS credentials to connect to [S3](https://aws.amazon.com/s3/) to be working. Ask @m.janvier for yours.

### Add a new data source

``` bash
dvc add path/to/file.extension
# The command generates the .dvc file
git add path/to/file.extension.dvc
git commit -m 'add data file'
dvc push
```

### Download data sources

``` bash
git checkout <commit>
dvc pull
```
The downloaded data sources will be the ones versionned at the commit you're currently at.
