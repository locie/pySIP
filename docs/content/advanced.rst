PySIP advanced usage and internals
==================================

This part of the documentation focus on what behind the scene of PySIP, to build
the intuition of how to use it and how to extend it.

Parameters and Priors
---------------------

The :class:`Parameter <pysip.params.Parameter>` class hold a parameter value with some extra
information : name, bounds, prior, scale...

At it's simplest form, a parameter is just a value and a name (the value is even
optional and default to 0.0):

.. code-block:: python

    >>> from pysip.params import Parameter
    >>> p = Parameter(name="p", value=1.0)
    >>> p
    Parameter(name='p', value=1.0, loc=0.0, scale=1.0, bounds=(None, None), transform=none, prior=None)

As you can see in the parameter representation, there is a lot of extra
information you can provide to the parameter. For example, you can specify
bounds and scale :

.. code-block:: python

    >>> p = Parameter(name="p", value=1.0, bounds=(0.0, 2.0), scale=0.1)
    >>> p
    Parameter(name='p', value=1.0, loc=0.0, scale=0.1, bounds=(0.0, 2.0), transform=logit, prior=None)

And something really interesting happend here. The transform has been
automatically set to :class:`logit <pysip.params.transforms.LogitTransform>` :
this transformation is used to contraint the parameter value to be in the
bounds. This lead to the next point : the constrained (\eta) parameter space and
the unconstrained (\theta) parameter space.

Constrained and unconstrained parameter space
+++++++++++++++++++++++++++++++++++++++++++++

The parameters are defined both in the unconstrained space and the constrained
space. The primer correspond to the :math:`\mathbb{R}`` space (all the real
numbers). At best, the parameter is standardized and lies around 0.0 with a
standard deviation of 1.0. Of course, this can be difficult to ensure for
unknown parameters.

The later correspond to the space where the parameter is constrained and that
are used in the model. For example, a parameter that is constrained to be
positive will have a constrained space of :math:`\mathbb{R}^+` (all the positive
real numbers).

The next section will explain how to define the transform between the two
spaces.

Transform and scale
+++++++++++++++++++

The transformation between the two spaces is done by a transformation function
then is rescaled by the scale parameter.

The transform is automatically set to a :class:`NoneTransform transform
<pysip.params.transforms.NoneTransform>` if no bounds are provided, to a
:class:`LogTransform <pysip.params.transforms.LogTransform>` if the lower bound
is 0.0 and to a :class:`UpperTransform <pysip.params.transforms.UpperTransform>`
(resp. :class:`LowerTransform <pysip.params.transforms.LowerTransform>`) if the
upper (resp lower) bound is set.

.. warning::
   The user can also provide a transform to the parameter : in that case, the bound
   could not be enforced depending on the transform ! Up to the user to be consistent
   between the bounds and the transform.

The scale is used to rescale the parameter value in the unconstrained space. It
default to 1.0, which means no rescaling is done.

.. code-block:: python

    >>> from pysip.params import Parameter
    >>> p = Parameter(name="p", value=1.0, scale=0.1)
    >>> p.theta, p.eta # theta is the unconstrained space, eta the constrained one
    (0.2, 2.0)

Why all this fuss about the constrained and unconstrained space ? Thanks to
that, the optimization algorithm can work in the unconstrained space, which is
much more easier to optimize and will naturally respect the bounds in the
constrained space instead of relying on a penalty function or using specialized
algorithms.

The :mod:`transforms <pysip.params.transforms>` API documentation will give you
more information about the available transforms and their mathematical
formulation.

In an alternative way, the `penalty` attribute can be used to define a penalty
function that will be used to penalize the parameter value if it is out of
bounds.

As an example of implementation, see :

.. code-block:: python

    from pysip.params.transforms import ParameterTransform, register_transform

    @register_transform
    class LogTransform(ParameterTransform):
        """Log transform, i.e. θ = exp(η)"""

        name = "log"

        def transform(self, θ: float) -> float:
            return np.log(θ)

        def untransform(self, η: float) -> float:
            return np.exp(η)

        def grad_transform(self, θ: float) -> float:
            return 1.0 / θ

        def grad_untransform(self, η: float) -> float:
            return np.exp(η)

        def penalty(self, θ: float) -> float:
            return 1e-12 / (θ - 1e-12)

        def grad_penalty(self, θ: float) -> float:
            return -1e-12 / (θ - 1e-12) ** 2

You can follow this template to implement your own transform. using the
`@register_transform` will automatically register the transform in the
transform registry, and make the transform available in the
:class:`pysip.params.Parameter` class.

Prior
+++++

The prior is a probability distribution that is used by the bayesian inference
as the prior knowledge about the parameter. It is used to compute the posterior
probability distribution of the parameter by updating this prior knowledge with
the likelihood of statespace estimator.

.. warning::
   The prior is defined in the constrained space. It will not be concerned by the
   transformation nor the scaling.

Good prior is heavily dependant on the problem at hand and expert knowledge. It
can be a uniform distribution between bounds if no prior knowledge is available,
or a gaussian around a value if the parameter is known to be close to this
value for example.

Prior are defined in the :mod:`priors <pysip.params.priors>` module. The API
documentation will give you more information about the available priors and
their mathematical formulation.

They hold both a :class:`scipy.stats.rv_continuous` as well as a
:class:`pymc.distributions.Continuous`, used for the frequentist and bayesian
inference respectively. An example of implementation is given here :

.. code-block:: python

    from pysip.params.priors import BasePrior, PriorMeta
    import pymc as pm
    from scipy import stats

    class Gamma(BasePrior, metaclass=PriorMeta):
        """Gamma prior distribution

        Parameters
        ----------
        alpha: float
            Shape parameter of the gamma distribution
        beta: float
            Rate parameter of the gamma distribution
        """

        alpha: float = 3.0
        beta: float = 1.0
        scipy_dist: lambda a, b: stats.gamma(a=a, scale=1.0 / b)
        pymc_dist: pm.Gamma

The `scipy_dist` attribute is used for the frequentist inference, and the
`pymc_dist` attribute is used for the bayesian inference. They take function
that will be called with the parameters of the prior as positional arguments to
create the distributions. Usually, you define the parameters to follow one of
the implementation and give a proxy to the other one, as in the example above.

Parameters collection and fixed parameters
++++++++++++++++++++++++++++++++++++++++++

Finally, a :class:`Parameters <pysip.params.Parameters>` collection hold a
collection of parameters : it is used to define the parameters of a model, and
their values can be updated at once by setting the :attr:`eta
<pysip.params.Parameters.eta>` or :attr:`theta <pysip.params.Parameters.theta>`
attribute.

.. code-block:: python

    >>> parameters = Parameters(
    ...     [
    ...         Parameter(name="a", value=1.0, bounds=(0.0, 10.0)),
    ...         Parameter(name="b", value=3.0, bounds=(0.0, None)),
    ...         Parameter(name="c", value=2.0, transform="fixed"),
    ...     ]
    ... )
    >>> parameters.theta = [1, 2, 3]
    >>> parameters
    Parameters
    Parameter(name='a', value=1.0, loc=0.0, scale=1.0, bounds=(0.0, 10.0), transform=logit, prior=None)
    Parameter(name='b', value=2.0, loc=0.0, scale=1.0, bounds=(0.0, None), transform=log, prior=None)
    Parameter(name='c', value=3.0, loc=0.0, scale=1.0, bounds=(None, None), transform=fixed, prior=None)
    >>> parameters.eta = [2, 1, 5]
    >>> parameters
    Parameters
    Parameter(name='a', value=8.807970779778824, loc=0.0, scale=1.0, bounds=(0.0, 10.0), transform=logit, prior=None)
    Parameter(name='b', value=2.718281828459045, loc=0.0, scale=1.0, bounds=(0.0, None), transform=log, prior=None)
    Parameter(name='c', value=2.0, loc=0.0, scale=1.0, bounds=(None, None), transform=fixed, prior=None)

You may have noticed the "fixed" transform. It is used to define a parameter
that are not supposed to be optimized.

That lead to the notion of fixed and free parameters : the later can be updated
using the `_free` suffix: :attr:`eta_free <pysip.params.Parameters.eta_free>` or
:attr:`theta_free <pysip.params.Parameters.theta_free>` attribute.

.. code-block:: python

    >>> parameters.theta_free = [1, 2]
    >>> parameters
    Parameters
    Parameter(name='a', value=1.0, loc=0.0, scale=1.0, bounds=(0.0, 10.0), transform=logit, prior=None)
    Parameter(name='b', value=2.0, loc=0.0, scale=1.0, bounds=(0.0, None), transform=log, prior=None)
    Parameter(name='c', value=2.0, loc=0.0, scale=1.0, bounds=(None, None), transform=fixed, prior=None)

Statespaces
-----------

The :class:`Statespace <pysip.statespace.Statespace>` class is the base class
that define a continuous model in PySIP. The models are defined using a
stochastic statespace representation as described here :


.. math::

    x_{t+1} = A x_t + B u_t + \mathcal{N}(0, Q)\\
    y = C x_t + D u_t + \mathcal{N}(0, R)

with :math:`x_t` the state vector, :math:`u_t` the input vector and :math:`y_t`
the observation vector.

:math:`A`, :math:`B`, :math:`C` and :math:`D` are the state, input, output and
feedthrough matrices respectively. :math:`Q` and :math:`R` are the state and
output covariance matrices.

The states are internal to the model and are not accessible to the user. The
input vector hold exogenous variables that are known at the time of the
prediction. The observation vector hold the output of the model that can be
compared to the real output to estimate the parameters.

As the model are defined in the continuous time, the statespace model has to be
discretized to be used in the optimization algorithm. The discretization is done
using the :meth:`StateSpace.discretization
<pysip.statespace.StateSpace.discretization>` method, that return discrete
matrices that will depend on the local timestep :math:`\Delta t`. A caching
strategy is used to avoid recomputing the discretization matrices at each
iteration for constant :math:`\Delta t`.

Diverse discretization method are available : it usually default to an analytic
discretization method if possible, and to a numerical method (MFD) otherwise.

By default, a zoh (zero order hold) discretization is used, but foh (first order
hold) is also available.

A large list of models are available in the :ref:`/model_list.rst` section.

To implement your model, you have can follow the template below :

.. code-block:: python

    from dataclasses import dataclass
    from pysip.statespace.base import RCModel

    @dataclass
    class Ti_RA(RCModel):

        states = [("TEMPERATURE", "xi", "indoor temperature")]

        params = [
            ("THERMAL_RESISTANCE", "R", "between the outdoor and the indoor"),
            ("THERMAL_CAPACITY", "C", "effective overall capacity"),
            ("SOLAR_APERTURE", "A", "effective solar aperture"),
            ("STATE_DEVIATION", "sigw", "of the indoor dynamic"),
            ("MEASURE_DEVIATION", "sigv", "of the indoor temperature measurements"),
            ("INITIAL_MEAN", "x0", "of the infoor temperature"),
            ("INITIAL_DEVIATION", "sigx0", "of the infoor temperature"),
        ]

        inputs = [
            ("TEMPERATURE", "To", "outdoor temperature"),
            ("POWER", "Qgh", "solar irradiance"),
            ("POWER", "Qh", "HVAC system heat"),
        ]

        outputs = [("TEMPERATURE", "xi", "indoor temperature")]

        def set_constant_continuous_ssm(self):
            self.C[0, 0] = 1.0

        def update_continuous_ssm(self):
            R, C, A, sigw, sigv, x0, sigx0, *_ = self.parameters.theta

            self.A[0, 0] = -1.0 / (C * R)
            self.B[0, :] = [1.0 / (C * R), A / C, 1.0 / C]
            self.Q[0, 0] = sigw
            self.R[0, 0] = sigv
            self.x0[0, 0] = x0
            self.P0[0, 0] = sigx0

in the `set_constant_continuous_ssm` method, you define the constant part of the
continuous matrices. In the `update_continuous_ssm` method, you define the
dynamic part of the continuous matrices that need to be updated at each
parameter changes.

Statespace estimator
--------------------

The Statespace Estimator is the class that hold the data and the model and that
is used to estimate the parameters. In pySIP, a Square Root Kalman filter is
used to estimate the states and the parameters.

It is written using :ref:`https://numba.pydata.org/` to speed up the computation
as much as possible.

The estimator hold the model (and its parameters) and will use provided data
to estimate and predict the states and their covariance. You can have a look at
the :ref:`/api.rst#statespace-estimators` for more detail on the available
methods.

Usually, users will not have to use the estimator directly, but will use the
:class:`pysip.regressors.Regressor` class that will handle the estimator and
the model for them.

Regressor
---------

The :class:`Regressor <pysip.regressors.Regressor>` class is the main class
that is used to estimate the parameters of a model. It hold the statespace, the
statespace estimator and give access to both frequentist and bayesian inference
methods.

.. code-block:: python

    import numpy as np
    import pandas as pd
    from pysip import Regressor
    from pysip.statespace import Matern32

    N = 50
    np.random.seed(1)
    t = np.linspace(0, 1, N)
    y = np.sin(12 * t) + 0.66 * np.cos(25 * t) + np.random.randn(N) * 0.25
    df = pd.DataFrame(index=pd.Index(t, name="t"), data=y, columns=["y"])

    par = [
        dict(name="mscale", value=5.11, bounds=(0, None)),
        dict(name="lscale", value=8.15, bounds=(0, None)),
        dict(name="sigv", value=0.8, bounds=(0, None)),
    ]
    model = Matern32(par)
    reg = Regressor(model, outputs=["y"])

You have to provide the outputs and inputs name that link the statespace model
to the data column names.

The regressor will not be heavily described here, as it is already the focus of
the :ref:`/content/quickstart.rst` section and well documented in the
:ref:`/cookbook.rst` section.
