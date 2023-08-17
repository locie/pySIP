# Stochastic State-Space Inference and Prediction in Python (pySIP)

Ordinary differential equations, determined from first-principles, can be used
for modeling physical systems. However, the exact dynamics of such systems are
uncertain and only measured at discrete-time instants through non-ideal sensors.
In this case, stochastic differential equations provide a modeling framework
which is more robust to these uncertainties. The stochastic part of the
state-space model can accommodate for unmodeled disturbances, which do not have
a significant influence on the system dynamics. Otherwise, unmeasured
disturbances can be modeled as temporal **Gaussian Processes** with certain
parametrized covariance structure. The resulting **Latent Force Model** is a
combination of parametric grey-box model and non-parametric Gaussian process
model.

**pySIP** provides a framework for **infering continuous time linear stochastic
state-space models**. For that purpose, it is possible to chose between a
**frequentist** and a **Bayesian** workflow. Each workflow allows to estimate
the parameters, assess the inference and model reliability, and perform model
selection.

**pySIP** is being developed in the perspective to build a library which gather
models from different engineering applications. Currently, applications
involving dynamic thermal models (RC network) and temporal Gaussian Processes
are being prioritized. Nevertheless, any model following the formalism of
**pySIP** can benefit from the features.

**pySIP** is currently under development and in beta version. Please feel free
to contact us if you want to be involved in the current development process.

You can find the documentation [here](https://locie.github.io/pySIP/) : it
contains a quick start guide, a cookbook, a tour of the
library internals and a reference documentation.

## IMPORTANT - Migration to v1.0.0

The version 1.0.0 of pySIP is a major update of the library. It focused on
(slightly) reducing the scope of the library and delegating some tasks to other
libraries (mainly scipy for the distribution, numba for the code acceleration
and pymc for the bayesian inference).

Regression are expected to happen, but the library should be in road to a more
stable state.

Main changes are:

- using [pymc3] for the bayesian inference (all the mcmc module have been
  removed)
- removing all analytical jacobian computation (using numerical approximation of
  the jacobian instead)
- full Regressor class rework : there is no more separation between the
  `FrequentistRegressor` and the `BayesianRegressor`. The regressor has now the
  ability to perform both frequentist (with the `regressor.fit` method) and
  bayesian (with the `regressor.sample` method) inference.
- the KalmanQR class is now accelerated using numba
- the library use `pandas.DataFrame` or `xarray.Dataset` as output for the
  `Regressor` class, allowing easier manipulation of the results.

Do not hesitate to open an issue if you encounter any problem with the new
version. If you encounter a regression, it should be possible to re-introduce
the functionality. When the version will be in the main branch, a gitter channel
will be created to help the migration.

A release tag is available for the previous version of the library (v0.9.0). It
will also be updated on pip : if you are not ready to migrate to the new
version, you can install the previous version using `pip install
git+https://github.com/locie/pySIP@v0.9.0`.

## Contributors

* [Loïc Raillon](https://github.com/LoicRaillon) - [Univ. Grenoble Alpes, Univ.
  Savoie Mont Blanc](https://www.locie.univ-smb.fr/en/home/), CNRS, LOCIE, 73000
Chambéry, France,
* [Maxime Janvier](https://github.com/mjanv) - [Lancey Energy
  Storage](https://www.lancey.fr/en/)
* [Nicolas Cellier](https://github.com/celliern) - [IMT Mines Albi -
  CGI](https://orcid.org/0000-0002-3759-3546) (current maintainer)

## Funding

* Auvergne Rhône-Alpes, project HESTIA-Diag, habitat Econome avec Système
Thermique Innovant Adapté pour le Diagnostic


[pymc3]: https://docs.pymc.io/