from bopt.utils.draw import TikzStateSpace


def test_constrained_model():

    class A(TikzStateSpace):
        inputs = ['T_0', 'Q_0']
        outputs = ['T_i']
        nodes = ['x_w']

        edges = [
            ('T_0', 'x_w', 'R_o'),
            ('g1', 'x_w', 'C_w'),
            ('T_i', 'x_w', 'R_i'),
            ('Q_0', 'x_w', ''),
        ]

        __tikz__ = {'T_0': (-2, 0), 'x_w': (0, 0), 'T_i': (2, 0), 'Q_0': (0, 1), 'g1': (0, -3)}

    assert '.. tikz::' in A().tikz


def test_handmade_model():

    class A(TikzStateSpace):
        """Handmade model"""

        __tikz__ = r"""
    .. tikz::

    \draw
    (0,0) to[american, V=$T_0$] (2,0) (0,0) node[ground, rotate=-90]{}
"""

    assert '.. tikz::' in A().tikz


def test_no_model():

    class A(TikzStateSpace):
        """No model"""

    assert '.. tikz::' not in A().tikz
