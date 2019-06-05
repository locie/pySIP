import math
import networkx as nx
import matplotlib.pyplot as plt


def find_angle(layout, start, end, unit=None):
    dx = layout[end][0] - layout[start][0]
    dy = layout[end][1] - layout[start][1]
    rad = math.atan2(dy, dx)
    angle = rad * (180 / math.pi)
    if not unit:
        return rad, angle
    if unit == 'radians':
        return rad
    return angle


class TikzStateSpace:

    __tikz__ = None
    __tikz_scale__ = 3
    __tikz_mode__ = 'spring'

    @property
    def tikz(self):
        canevas = r"""
    .. tikz::

        \ctikzset{{label/align = straight}}
        \ctikzset{{resistor = european}}
        \draw
        {}
        ;

        """

        if self.__tikz__ is None:
            return ''

        if isinstance(self.__tikz__, str):
            if '.. tikz::' in self.__tikz__:
                return self.__tikz__
            return canevas.format(self.__tikz__)

        G = nx.Graph()
        G.add_nodes_from(self.inputs + self.outputs + self.nodes)
        for start, end, metadata in self.edges:
            G.add_edge(start, end, metadata=metadata)

        if self.__tikz_mode__ == 'spring':
            if isinstance(self.__tikz__, dict):
                pos = {k: v for k, v in self.__tikz__.items() if k in self.inputs + self.outputs + self.nodes}
                fixed = pos.keys() if pos else None
            else:
                pos, fixed = None, None
            layout = nx.drawing.layout.spring_layout(G, scale=self.__tikz_scale__, pos=pos, fixed=fixed)
        else:
            layout = nx.drawing.layout.planar_layout(G, scale=self.__tikz_scale__)
        layout = {k: tuple(int(x) for x in v) for k, v in layout.items()}

        lines = []

        for n in self.nodes:
            if not n.startswith('g'):
                lines.append(f'\t{layout[n]} node[above left] {{ ${n}$ }}')
                lines.append(f'\t{layout[n]} node[circle,fill,scale=0.2] {{ }}')

        for i in self.inputs + self.outputs:
            t = 'V' if i.startswith('T') else 'I'

            neighbors = list(filter(lambda x: x[0] == i, self.edges))
            a, b = (neighbors[0][:2]) if neighbors else (None, None)

            if a and b:
                rad, angle = find_angle(layout, a, b)
            else:
                rad, angle = math.pi / 2, 90

            start = layout[i][0] - 2 * math.cos(rad), layout[i][1] - 2 * math.sin(rad)

            lines.append(f'{start} to[american, {t}=${i}$] {layout[i]}')
            lines.append(f'{start} node[ground, rotate={int(angle)-90}]{{}}')

        for start, end, metadata in self.edges:
            middle = '--'
            if metadata.startswith('R') or metadata.startswith('\\frac{R'):
                middle = f'to[R=${metadata}$]'
            if metadata.startswith('C'):
                middle = f'to[C=${metadata}$]'

            if isinstance(self.__tikz__, dict) and metadata in self.__tikz__ and metadata not in self.inputs + self.outputs + self.nodes:
                pos = self.__tikz__[metadata]
                s, e = layout[start], layout[end]
                N = 2
                a = pos[0] - (pos[0] - s[0]) / N, pos[1] - (pos[1] - s[1]) / N
                b = pos[0] + (e[0] - pos[0]) / N, pos[1] + (e[1] - pos[1]) / N
                lines.append(f'{s} -- {a} {a} {middle} {b} {b} -- {e}')
            else:
                lines.append(f'{layout[start]} {middle} {layout[end]}')

            if start == '' or start.startswith('g'):
                angle = find_angle(layout, start, end, 'degrees')
                lines.append(f'{layout[start]} node[ground, rotate={int(angle)-90}]{{}}')

        return canevas.format('\n'.join('\t' + l for l in lines))
