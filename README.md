# acoucalc

This library is ment to provide a simple way to calculate acoustic 
behaviour of multilayered absorbers using the transfer matrix method, 
as described in [Allard and Atalla (2009)](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470747339).

So far, it is possible to calculate the surface impedance and the absorption using the Johnson-Champoux-Allard model as well as the more simple Delany-Bazley model.

## Installation

It is currently not possible to install this library using `pip` or `conda`, please use the latest [released package](https://github.com/vyhyb/acoucalc/releases) instead and install using [`pip` locally](https://packaging.python.org/en/latest/tutorials/installing-packages/).

## Documentation

Documentation can be found [here](https://vyhyb.github.io/acoucalc/).

## Usage

The following example shows how to calculate the absorption coefficient of a simple absorber using the JCA model.

```python
import numpy as np
import matplotlib.pyplot as plt

from acoucalc.models import air, jca
from acoucalc.layers import tm_fluid
from acoucalc.core import (
    initial_pv,
    add_layer, 
    surface_impedance, 
    pressure_refl_factor, 
    absorption_coefficient
)

freqs = np.linspace(63, 4000, 1000)
thichness_air = 0.1
thickness_porous = 0.05

pv_init = initial_pv(1, 0, freqs)

effective_density_air, bulk_modulus_air = air(freqs)
tm = tm_fluid(thichness_air, effective_density_air, bulk_modulus_air, freqs)
pv_air = add_layer(pv_init, tm)

eff_density, bulk_modulus = jca(
        flow_resistivity=12000,
        porosity=0.95,
        tortuosity=1.1,
        viscous_char_length=100e-6,
        thermal_char_length=200e-6,
        f=freqs
    )

tm = tm_fluid(thickness_porous, eff_density, bulk_modulus, freqs)
pv = add_layer(pv_air, tm)
z = surface_impedance(pv)
r = pressure_refl_factor(z)
alpha = absorption_coefficient(r)

plt.semilogx(freqs, alpha)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Absorption coefficient')
plt.show()
```

## Acknowledgments

This library was created thanks to the [FAST-S-24-8572](https://www.vut.cz/vav/projekty/detail/36174) project.

Github Copilot was used to generate parts of the documentation and code.

## Author

- [David Jun](https://www.fce.vutbr.cz/o-fakulte/lide/david-jun-12801/)
  
  PhD student at [Brno University of Technology](https://www.vutbr.cz/en/) and [KU Leuven](https://www.kuleuven.be/english/)

## Contributing

Pull requests are welcome. For any changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)

## References

- [1] J.-F. Allard and N. Atalla, Propagation of sound in porous media, 2nd ed. Wiley, 2009.
- [2] T. J. Cox and P. D’Antonio, Acoustic absorbers and diffusers: Theory, design and application, 3rd ed. Taylor, 2017. doi: 10.4324/9781482266412.
