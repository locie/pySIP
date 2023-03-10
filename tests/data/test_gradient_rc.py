import numdifftools as nd
import numpy as np
import pandas as pd
import pytest

from pysip.core import Beta, Gamma, InverseGamma, LogNormal, Normal
from pysip.regressors import FreqRegressor as Regressor
from pysip.statespace import model_registry
from pysip.utils import generate_random_binary, generate_sine, ned

rc_models = {
    k: v
    for k, v in model_registry.items()
    if "pysip.statespace.thermal_network" in v.__module__
}


@pytest.fixture
def artificial_data_rc():
    """Generate 50 data points spaced of 1 hour

    Ti: Indoor temperature
    To: Outdoor temperature
    Tb: Boundary temperature
    Qgh: Global horizontal solar radiation
    Qh: Heat supplied by heaters
    Qv: Heat supplied by ventilation
    Ql: Heat supplied by internal load
    """
    n = 50  # number of points
    dt = 3600  # sampling time
    df = pd.DataFrame(
        {
            "Ti": generate_sine(n=n, amplitude=3.0, phase=np.pi / 3.0, offset=20.0)[1],
            "To": generate_sine(n=n)[1],
            "Tb": generate_sine(n=n, amplitude=2.0, phase=np.pi / 4.0, offset=14.0)[1],
            "Qgh": generate_sine(n=n, amplitude=800.0, clip_to_0=True)[1],
            "Qh": generate_random_binary(n=n),
            "Qv": generate_sine(n=n, amplitude=200.0, offset=100.0)[1],
            "Ql": 100.0 + 25.0 * np.random.randn(n),
        }
    )
    df.index *= dt
    return df


def random_parameters(statespace):
    parameters = []
    transforms = ["log", "lower", "upper", "logit"]
    priors = [Normal(), LogNormal(), InverseGamma(), Gamma(), Beta()]
    scale_dict = {
        "THERMAL_RESISTANCE": 1e-2,
        "THERMAL_TRANSMITTANCE": 1e2,
        "THERMAL_CAPACITY": 1e8,
        "SOLAR_APERTURE": 1.0,
        "COEFFICIENT": 1.0,
        "STATE_DEVIATION": 1e-2,
        "MEASURE_DEVIATION": 1e-2,
        "INITIAL_MEAN": 1e2,
        "INITIAL_DEVIATION": 1.0,
    }
    ntrfm = len(transforms) - 1
    nprior = len(priors) - 1

    for _par in statespace.params:
        category, name, _ = _par
        parameters.append(
            dict(
                name=name,
                value=np.random.uniform(0.1, 0.9),
                scale=scale_dict[category],
                transform=transforms[int(np.round(np.random.uniform(high=ntrfm)))],
                bounds=(0.001, 1.0),
                prior=priors[int(np.round(np.random.uniform(high=nprior)))],
            )
        )

    return parameters


def get_inputs(statespace):
    """Create list of inputs for the regressor"""

    inputs = []
    if not statespace.inputs == []:
        for _input in statespace.inputs:
            name = _input[1]
            if name in ["To", "Tb", "Qgh", "Qh", "Qv", "Ql"]:
                inputs.append(name)
            else:
                raise ValueError("Unknown input name")
    return inputs


def get_outputs(statespace):
    """Create list of outputs for the regressor"""
    if len(statespace.outputs) > 1:
        raise NotImplementedError
    _outputs = "Ti"

    return _outputs


def gen_regressor(statespaces):
    """State-space generator for gradient test"""
    for ssm in statespaces:
        p = random_parameters(ssm)
        inputs = get_inputs(ssm)
        outputs = get_outputs(ssm)
        h = np.round(np.random.rand())

        yield Regressor(ss=ssm(parameters=p, hold_order=h)), inputs, outputs


@pytest.mark.parametrize(
    "reg, inputs, outputs", gen_regressor([m for m in rc_models.values()])
)
def test_gradient_RCModel(artificial_data_rc, reg, inputs, outputs):
    """Compare regressor gradient with numerical differentiation"""
    reg._use_penalty = False
    reg._use_jacobian = True

    dt, u, u1, y, *_ = reg._prepare_data(artificial_data_rc, inputs, outputs)
    eta = reg.ss.parameters.eta_free.copy()
    grad = reg._eval_dlog_posterior(eta, dt, u, u1, y)[1]

    fct = nd.Gradient(reg._eval_log_posterior)
    grad_fd = fct(eta, dt, u, u1, y)

    assert ned(grad, grad_fd) < 1e-7
    assert np.all(np.sign(grad) == np.sign(grad_fd))
    assert grad == pytest.approx(grad_fd, rel=1e-3)
