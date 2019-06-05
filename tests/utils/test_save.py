from bopt.regressors import Regressor
from bopt.statespace import TwTi_RoRi
from bopt.utils import save_model, load_model


def test_save():
    reg = Regressor(TwTi_RoRi())
    save_model('test', reg)
    load_reg = load_model('test')

    assert id(reg) != id(load_reg)

    for a, b in zip(reg.ss.parameters, load_reg.ss.parameters):
        assert a == b
