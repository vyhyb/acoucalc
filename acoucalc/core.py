"""A module providing the core functionality of the acoucalc package.
"""
import numpy as np

def add_layer(
        pv: np.ndarray, 
        tm: np.ndarray
    ) -> np.ndarray:
    """Add a layer to the existing system.

    Parameters
    ----------
    pv : np.ndarray
        Pressure and velocity vectors.
    tm : np.ndarray
        Transfer matrix of the new layer.

    Returns
    -------
    np.ndarray
        Updated pressure and velocity vectors.
    """
    pv_new = np.einsum('ijk,jk->ik', tm, pv)
    return pv_new

def initial_pv(
        p: float|np.ndarray, 
        v: float|np.ndarray, 
        f: np.ndarray=None
    ) -> np.ndarray:
    """Create the initial pressure and velocity vectors
    based on the boundary conditions.

    Parameters
    ----------
    p : float | np.ndarray
        Pressure at the boundary.
    v : float | np.ndarray
        Velocity at the boundary.
    f : np.ndarray, optional
        Frequency vector, by default None.

    Returns
    -------
    np.ndarray
        Initial pressure and velocity vectors.
    """
    if f is not None:
        p = p*np.ones_like(f)
        v = v*np.ones_like(f)
    pv = np.array([p, v])
    return pv

def surface_impedance(pv: np.ndarray) -> np.ndarray:
    """Calculate the surface impedance of the system.

    Parameters
    ----------
    pv : np.ndarray
        Pressure and velocity vectors.

    Returns
    -------
    np.ndarray
        Surface impedance.
    """
    return pv[0]/pv[1]

def pressure_refl_factor(
        surf_impedance:np.ndarray,
        char_impedance:float|np.ndarray=343*1.21
    ) -> np.ndarray:
    return (surf_impedance-char_impedance)/(surf_impedance+char_impedance)

def absorption_coefficient(refl_factor:np.ndarray) -> np.ndarray:
    """Calculate the absorption coefficient from the reflection factor.

    Parameters
    ----------
    refl_factor : np.ndarray
        Reflection factor.

    Returns
    -------
    np.ndarray
        Absorption coefficient.
    """
    return 1 - np.abs(refl_factor)**2

def transmission_coefficient(
        tm: np.ndarray,
        char_impedance_air: float
    ) -> np.ndarray:
    """Calculate the transmission coefficient of the system.

    Parameters
    ----------
    tm : np.ndarray
        Global transfer matrix of the system.
    char_impedance_air : float
        Characteristic impedance of air.

    Returns
    -------
    np.ndarray
        Transmission coefficient.
    """
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

def reflected_pressure(
        p_init: float|np.ndarray,
        tm: np.ndarray, 
        char_impedance_air: float
    ) -> np.ndarray:
    """Calculate the reflected pressure from the system.

    Parameters
    ----------
    p_init : float | np.ndarray
        Initial pressure.
    tm : np.ndarray
        Global transfer matrix of the system.
    char_impedance_air : float
        Characteristic impedance of air.

    Returns
    -------
    np.ndarray
        Reflected pressure.
    """
    A = tm[0][0] + tm[0][1]/char_impedance_air
    B = tm[1][0]*char_impedance_air + tm[1][1]
    p_refl = p_init * (A - B) / (A + B)
    return p_refl

def transmitted_pressure(
        p_init: float|np.ndarray,
        tm: np.ndarray,
        char_impedance_air: float,
    ) -> np.ndarray:
    """Calculate the pressure transmitted through the system.

    Parameters
    ----------
    p_init : float | np.ndarray
        Initial pressure.
    tm : np.ndarray
        Global transfer matrix of the system.
    char_impedance_air : float
        Characteristic impedance of air.

    Returns
    -------
    np.ndarray
        Pressure transmitted through the system.
    """
    p_tr = p_init * transmission_coefficient(tm, char_impedance_air)
    return p_tr

