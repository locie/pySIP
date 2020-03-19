from dataclasses import dataclass, field
from enum import Enum
from typing import Union


class Io(Enum):
    """Input/Output data type"""

    TEMPERATURE = ('Temperature', '°C')
    POWER = ('Power', 'W')
    ANY = ('Any type', 'any')


class Par(Enum):
    """Parameter type"""

    THERMAL_RESISTANCE = ('Thermal resistance', '°C/W')
    THERMAL_TRANSMITTANCE = ('Thermal transmittance', 'W/°C')
    THERMAL_CAPACITY = ('Thermal capacity', 'J/°C')
    SOLAR_APERTURE = ('Solar aperture', 'm²')
    STATE_DEVIATION = ('State deviation', 'any')
    MEASURE_DEVIATION = ('Measure deviation', 'any')
    INITIAL_DEVIATION = ('Initial deviation', 'any')
    INITIAL_MEAN = ('Initial mean', 'any')
    COEFFICIENT = ('Coefficient', 'any')
    MAGNITUDE_SCALE = ('Magnitude scale', 'any')
    LENGTH_SCALE = ('Length scale', 'any')
    PERIOD = ('Period', 'any')


@dataclass
class Node:
    """Description of variables"""

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
