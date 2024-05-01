"""Module for transfer matrices of different layers
"""
import numpy as np

# transfer matrix for fluid layer
def tm_fluid(thickness, eff_density, bulk_modulus, f):
    w = 2*np.pi*f
    wavenumber = w*np.sqrt(eff_density/bulk_modulus)

    # tranfer matrix members for fluid layer
    t11 = np.cos(wavenumber*thickness)
    t12 = 1j*(w*eff_density/wavenumber)*np.sin(wavenumber*thickness)
    t21 = 1j*(wavenumber/eff_density/w)*np.sin(wavenumber*thickness)
    t22 = np.cos(wavenumber*thickness)
    # transfer matrix
    tm = np.array([[t11, t12],[t21, t22]])
    return tm

def tm_thin_plate(impedance):
    t11 = np.ones_like(impedance)
    t12 = -impedance
    t21 = np.zeros_like(impedance)
    t22 = np.ones_like(impedance)
    return np.array([[t11, t12],[t21, t22]])

def tm_perforation(impedance):
    t11 = np.ones_like(impedance)
    t12 = impedance
    t21 = np.zeros_like(impedance)
    t22 = np.ones_like(impedance)
    return np.array([[t11, t12],[t21, t22]])
