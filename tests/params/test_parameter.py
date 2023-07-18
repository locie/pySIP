from collections import namedtuple

import pytest
from pydantic import ValidationError

from pysip.params import Parameter
from pysip.params.prior import Normal

defaults = namedtuple("Defaults", "name value scale bounds prior theta")(
    name="name",
    value=0.6,
    scale=0.5,
    bounds=(0.0, 1.0),
    prior=Normal(0, 1),
    theta=0.6 * 0.5,
)


def parameter_fixture(transform, **kwargs):
    kwargs = {**{"transform": transform}, **defaults._asdict(), **kwargs}
    parameter = Parameter(**kwargs)
    def fixture() -> Parameter:
        return parameter
    return pytest.fixture(fixture)


hp_none = parameter_fixture("none")
hp_log = parameter_fixture("log")
hp_logit = parameter_fixture("logit")
hp_fixed = parameter_fixture("fixed")
hp_lower = parameter_fixture("lower", bounds=(0.1, None))
hp_upper = parameter_fixture("upper", bounds=(None, 0.9))


def test_init_none(hp_none: Parameter):
    assert hp_none.name == defaults.name
    assert hp_none.theta == defaults.theta
    assert hp_none.value == defaults.value
    assert hp_none.scale == defaults.scale
    assert hp_none.bounds == defaults.bounds
    assert hp_none.prior == defaults.prior


def test_init_log(hp_log: Parameter):
    assert hp_log.name == defaults.name
    assert hp_log.theta == defaults.theta
    assert hp_log.value == defaults.value
    assert hp_log.scale == defaults.scale
    assert hp_log.bounds == defaults.bounds
    assert hp_log.prior == defaults.prior


def test_init_logit(hp_logit: Parameter):
    assert hp_logit.name == defaults.name
    assert hp_logit.theta == defaults.theta
    assert hp_logit.value == defaults.value
    assert hp_logit.scale == defaults.scale
    assert hp_logit.bounds == defaults.bounds
    assert hp_logit.prior == defaults.prior


def test_init_fixed(hp_fixed: Parameter):
    assert hp_fixed.name == defaults.name
    assert hp_fixed.theta == defaults.theta
    assert hp_fixed.value == defaults.value
    assert hp_fixed.scale == defaults.scale
    assert hp_fixed.bounds == defaults.bounds
    assert hp_fixed.prior == defaults.prior


def test_init_lower(hp_lower: Parameter):
    assert hp_lower.name == defaults.name
    assert hp_lower.theta == defaults.theta
    assert hp_lower.value == defaults.value
    assert hp_lower.scale == defaults.scale
    assert hp_lower.bounds == (0.1, None)
    assert hp_lower.prior == defaults.prior


def test_init_upper(hp_upper: Parameter):
    assert hp_upper.name == defaults.name
    assert hp_upper.theta == defaults.theta
    assert hp_upper.value == defaults.value
    assert hp_upper.scale == defaults.scale
    assert hp_upper.bounds == (None, 0.9)
    assert hp_upper.prior == defaults.prior


def test_parameter_init_error_name_is_not_a_string():
    with pytest.raises(ValidationError):
        Parameter(None, None)


def test_parameter_init_error_lower_bound_gt_upper_bound():
    with pytest.raises(ValueError):
        Parameter(defaults.name, defaults.value, bounds=defaults.bounds[::-1])


def test_parameter_init_error_value_should_be_a_float():
    with pytest.raises(ValidationError):
        Parameter(defaults.name, complex(defaults.value))


def test_parameter_init_error_unvalid_transform():
    with pytest.raises(ValidationError):
        Parameter(defaults.name, defaults.value, transform="_")


def test_parameter_init_error_unvalid_prior():
    with pytest.raises(ValueError):
        Parameter(defaults.name, defaults.value, prior="_")


def test_parameter_none_transform(hp_none: Parameter):
    assert hp_none.value == defaults.value
    assert hp_none.theta == defaults.theta
    assert hp_none.eta == defaults.value
    assert hp_none.get_transform_jacobian() == 1.0
    assert hp_none.value == defaults.value
    assert hp_none.get_inv_transform_jacobian() == 1.0
    assert hp_none.get_penalty() == 0.0
    assert hp_none.get_grad_penalty() == 0.0


def test_parameter_fixed_transform(hp_fixed: Parameter):
    assert hp_fixed.value == defaults.value
    assert hp_fixed.theta == defaults.theta
    assert hp_fixed.eta == defaults.value
    assert hp_fixed.get_transform_jacobian() == 1.0
    assert hp_fixed.value == defaults.value
    assert hp_fixed.get_inv_transform_jacobian() == 1.0
    assert hp_fixed.get_penalty() == 0.0
    assert hp_fixed.get_grad_penalty() == 0.0


def test_parameter_log_transform(hp_log: Parameter):
    assert hp_log.value == defaults.value
    assert hp_log.theta == defaults.theta
    assert hp_log.eta == pytest.approx(-0.5108256)
    assert hp_log.get_transform_jacobian() == pytest.approx(1.666667)
    assert hp_log.value == pytest.approx(defaults.value)
    assert hp_log.get_inv_transform_jacobian() == defaults.value
    assert hp_log.get_penalty() == pytest.approx(1.6666666666694444e-12)
    assert hp_log.get_grad_penalty() == pytest.approx(-2.777777777787037e-12)


def test_parameter_logit_transform(hp_logit: Parameter):
    assert hp_logit.value == defaults.value
    assert hp_logit.theta == defaults.theta
    assert hp_logit.eta == pytest.approx(0.405465)
    assert hp_logit.get_transform_jacobian() == pytest.approx(4.166666)
    assert hp_logit.value == pytest.approx(defaults.value)
    assert hp_logit.get_inv_transform_jacobian() == pytest.approx(0.24)
    assert hp_logit.get_penalty() == pytest.approx(2.5)
    assert hp_logit.get_grad_penalty() == pytest.approx(6.249999)


def test_parameter_lower_transform(hp_lower: Parameter):
    assert hp_lower.value == defaults.value
    assert hp_lower.theta == defaults.theta
    assert hp_lower.eta == pytest.approx(-0.693147)
    assert hp_lower.get_transform_jacobian() == pytest.approx(2.0)
    assert hp_lower.value == pytest.approx(defaults.value)
    assert hp_lower.get_inv_transform_jacobian() == pytest.approx(0.5)
    assert hp_lower.get_penalty() == pytest.approx(0.2)
    assert hp_lower.get_grad_penalty() == pytest.approx(-0.4)


def test_parameter_upper_transform(hp_upper: Parameter):
    assert hp_upper.value == defaults.value
    assert hp_upper.theta == defaults.theta
    assert hp_upper.eta == pytest.approx(-1.203972)
    assert hp_upper.get_transform_jacobian() == pytest.approx(-3.333333)
    assert hp_upper.value == pytest.approx(defaults.value)
    assert hp_upper.get_inv_transform_jacobian() == pytest.approx(-0.3)
    assert hp_upper.get_penalty() == pytest.approx(2.999999)
    assert hp_upper.get_grad_penalty() == pytest.approx(9.999999)


def test_infer_transform_from_bounds():
    assert Parameter("", bounds=(None, None)).transform.name == "none"
    assert Parameter("", bounds=(0.0, None)).transform.name == "log"
    assert Parameter("", bounds=(5.0, None)).transform.name == "lower"
    assert Parameter("", bounds=(None, 5.0)).transform.name == "upper"
    assert Parameter("", bounds=(0.0, 5.0)).transform.name == "logit"
