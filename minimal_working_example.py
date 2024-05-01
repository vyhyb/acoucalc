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