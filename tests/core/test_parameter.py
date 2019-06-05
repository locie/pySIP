from collections import namedtuple

import pytest

from bopt.core import Parameter, Normal


defaults = namedtuple('Defaults', 'name value bounds prior')(
    name='name',
    value=0.6,
    bounds=(0, 1),
    prior=Normal(0, 1)
)


def parameter_fixture(transform, **kwargs):
    kwargs = {**{'transform': transform}, **defaults._asdict(), **kwargs}
    parameter = Parameter(**kwargs)
    return pytest.fixture(lambda: parameter)


hp_none = parameter_fixture('none')
hp_log = parameter_fixture('log')
hp_lowup = parameter_fixture('lowup')
hp_fixed = parameter_fixture('fixed')
hp_lower = parameter_fixture('lower', bounds=(0.1, None))
hp_upper = parameter_fixture('upper', bounds=(None, 0.9))


def test_init_none(hp_none):
    assert hp_none.name == defaults.name
    assert hp_none.value == defaults.value
    assert hp_none.bounds == defaults.bounds
    assert hp_none.prior == defaults.prior


def test_init_log(hp_log):
    assert hp_log.name == defaults.name
    assert hp_log.value == defaults.value
    assert hp_log.bounds == defaults.bounds
    assert hp_log.prior == defaults.prior


def test_init_lowup(hp_lowup):
    assert hp_lowup.name == defaults.name
    assert hp_lowup.value == defaults.value
    assert hp_lowup.bounds == defaults.bounds
    assert hp_lowup.prior == defaults.prior


def test_init_fixed(hp_fixed):
    assert hp_fixed.name == defaults.name
    assert hp_fixed.value == defaults.value
    assert hp_fixed.bounds == defaults.bounds
    assert hp_fixed.prior == defaults.prior


def test_init_lower(hp_lower):
    assert hp_lower.name == defaults.name
    assert hp_lower.value == defaults.value
    assert hp_lower.bounds == (0.1, None)
    assert hp_lower.prior == defaults.prior


def test_init_upper(hp_upper):
    assert hp_upper.name == defaults.name
    assert hp_upper.value == defaults.value
    assert hp_upper.bounds == (None, 0.9)
    assert hp_upper.prior == defaults.prior


def test_parameter_init_error_name_is_not_a_string():
    with pytest.raises(TypeError):
        Parameter(None, None)


def test_parameter_init_error_lower_bound_gt_upper_bound():
    with pytest.raises(ValueError):
        Parameter(defaults.name, defaults.value, bounds=defaults.bounds[::-1])


def test_parameter_init_error_value_should_be_a_float():
    with pytest.raises(TypeError):
        Parameter(defaults.name, int(defaults.value))


def test_parameter_init_error_value_should_be_a_float():
    with pytest.raises(TypeError):
        Parameter(defaults.name, int(defaults.value))


def test_parameter_init_error_unvalid_transform():
    with pytest.raises(ValueError):
        Parameter(defaults.name, defaults.value, transform='_')


def test_parameter_init_error_unvalid_prior():
    with pytest.raises(ValueError):
        Parameter(defaults.name, defaults.value, prior='_')


def test_parameter_none_transform(hp_none):
    assert hp_none.value == defaults.value
    assert hp_none._transform() == defaults.value
    assert hp_none._transform_jacobian() == 1.0

    hp_none._inv_transform(hp_none._transform())

    assert hp_none.value == defaults.value
    assert hp_none._inv_transform_jacobian() == 1.0
    assert hp_none._inv_transform_dlog_jacobian() == 0.0
    assert hp_none._inv_transform_d2log_jacobian() == 0.0
    assert hp_none._penalty() == 0.0
    assert hp_none._d_penalty() == 0.0


def test_parameter_fixed_transform(hp_fixed):
    assert hp_fixed.value == defaults.value
    assert hp_fixed._transform() == defaults.value
    assert hp_fixed._transform_jacobian() == 1.0

    hp_fixed._inv_transform(hp_fixed._transform())

    assert hp_fixed.value == defaults.value
    assert hp_fixed._inv_transform_jacobian() == 1.0
    assert hp_fixed._inv_transform_dlog_jacobian() == 0.0
    assert hp_fixed._inv_transform_d2log_jacobian() == 0.0
    assert hp_fixed._penalty() == 0.0
    assert hp_fixed._d_penalty() == 0.0


def test_parameter_log_transform(hp_log):
    assert hp_log.value == defaults.value
    assert hp_log._transform() == pytest.approx(-0.5108256)
    assert hp_log._transform_jacobian() == pytest.approx(1.666667)

    hp_log._inv_transform(hp_log._transform())

    assert hp_log.value == pytest.approx(defaults.value)
    assert hp_log._inv_transform_jacobian() == defaults.value
    assert hp_log._inv_transform_dlog_jacobian() == 1.0
    assert hp_log._inv_transform_d2log_jacobian() == 0.0
    assert hp_log._penalty() == pytest.approx(3.70074341541719e-16)
    assert hp_log._penalty() == pytest.approx(-6.167905692361985e-16)


def test_parameter_lowup_transform(hp_lowup):
    assert hp_lowup.value == defaults.value
    assert hp_lowup._transform() == pytest.approx(0.405465)
    assert hp_lowup._transform_jacobian() == pytest.approx(4.166666)

    hp_lowup._inv_transform(hp_lowup._transform())

    assert hp_lowup.value == pytest.approx(defaults.value)
    assert hp_lowup._inv_transform_jacobian() == pytest.approx(0.24)
    assert hp_lowup._inv_transform_dlog_jacobian() == pytest.approx(-0.1999999)
    assert hp_lowup._inv_transform_d2log_jacobian() == pytest.approx(-0.4799999)
    assert hp_lowup._penalty() == pytest.approx(2.5)
    assert hp_lowup._d_penalty() == pytest.approx(6.249999)


def test_parameter_lower_transform(hp_lower):
    assert hp_lower.value == defaults.value
    assert hp_lower._transform() == pytest.approx(-0.693147)
    assert hp_lower._transform_jacobian() == pytest.approx(2.0)

    hp_lower._inv_transform(hp_lower._transform())

    assert hp_lower.value == pytest.approx(defaults.value)
    assert hp_lower._inv_transform_jacobian() == defaults.value
    assert hp_lower._inv_transform_dlog_jacobian() == 1.0
    assert hp_lower._inv_transform_d2log_jacobian() == 0.0
    assert hp_lower._penalty() == pytest.approx(0.2)
    assert hp_lower._d_penalty() == pytest.approx(-0.4)


def test_parameter_upper_transform(hp_upper):
    assert hp_upper.value == defaults.value
    assert hp_upper._transform() == pytest.approx(-1.203972)
    assert hp_upper._transform_jacobian() == pytest.approx(-3.333333)

    hp_upper._inv_transform(hp_upper._transform())

    assert hp_upper.value == pytest.approx(defaults.value)
    assert hp_upper._inv_transform_jacobian() == defaults.value
    assert hp_upper._inv_transform_dlog_jacobian() == 1.0
    assert hp_upper._inv_transform_d2log_jacobian() == 0.0
    assert hp_upper._penalty() == pytest.approx(2.999999)
    assert hp_upper._d_penalty() == pytest.approx(9.999999)


def test_infer_transform_from_bounds():
    assert Parameter('', bounds=(None, None)).transform == 'none'
    assert Parameter('', bounds=(0, None)).transform == 'log'
    assert Parameter('', bounds=(5, None)).transform == 'lower'
    assert Parameter('', bounds=(None, 5)).transform == 'upper'
    assert Parameter('', bounds=(0, 5)).transform == 'lowup'
