"""A module providing the core functionality of the acoucalc package.
"""
import numpy as np

def add_layer(pv, tm):
    pv_new = np.einsum('ijk,jk->ik', tm, pv)
    return pv_new

def initial_pv(p, v, f=None):
    if f is not None:
        p = p*np.ones_like(f)
        v = v*np.ones_like(f)
    pv = np.array([p, v])
    return pv

def surface_impedance(pv):
    return pv[0]/pv[1]

def pressure_refl_factor(surf_impedance, char_impedance=343*1.21):
    return (surf_impedance-char_impedance)/(surf_impedance+char_impedance)

def absorption_coefficient(refl_factor):
    return 1 - np.abs(refl_factor)**2

def transmission_coefficient(tm, char_impedance_air, wavenumber_air, thickness):
    tc = (
        # np.exp(1j * wavenumber_air * thickness) *
        2 /
        (tm[0][0] 
            + tm[0][1]/char_impedance_air
            + char_impedance_air*tm[1][0]
            + tm[1][1]
        )
    )
    return tc

def reflected_pressure(p_init, tm, char_impedance_air):
    A = tm[0][0] + tm[0][1]/char_impedance_air
    B = tm[1][0]*char_impedance_air + tm[1][1]
    p_refl = p_init * (A - B) / (A + B)
    return p_refl

def transmitted_pressure(p_init, tm, char_impedance_air, wavenumber_air, thickness):
    p_tr = p_init * transmission_coefficient(tm, char_impedance_air, wavenumber_air, thickness)
    return p_tr

