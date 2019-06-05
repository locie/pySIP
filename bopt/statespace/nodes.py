from dataclasses import dataclass, field
from enum import Enum
from typing import Union


class Io(Enum):
    TEMPERATURE = ('Temperature', '°C')
    POWER = ('Power', 'W')
    ANY = ('Any type', '')


class Par(Enum):
    THERMAL_RESISTANCE = ('Thermal resistance', 'K/W')
    THERMAL_CAPACITY = ('Thermal capacity', 'J/K')
    SOLAR_APERTURE = ('Solar aperture', 'm2')
    STATE_DEVIATION = ('State deviation', '')
    MEASURE_DEVIATION = ('Measure deviation', '')
    INITIAL_DEVIATION = ('Initial deviation', '')
    INITIAL_MEAN = ('Initial mean', '')
    COEFFICIENT = ('Coefficient', '')
    MAGNITUDE_SCALE = ('Magnitude scale', '')
    LENGTH_SCALE = ('Length scale', '')
    PERIOD = ('Period', '')


@dataclass
class Node:
    category: Union[Io, Par] = field()
    name: str = field(default='')
    description: str = field(default='')

    def __post_init__(self):
        if isinstance(self.category, str):
            try:
                self.category = Io[self.category]
            except KeyError:
                self.category = Par[self.category]

    def unpack(self):
        '''Unpack the Node'''
        return self.category.name, self.name, self.description
