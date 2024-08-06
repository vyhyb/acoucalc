"""Module for transfer matrices of different layers
"""
import numpy as np

# transfer matrix for fluid layer
def tm_fluid(
        thickness: float,
        eff_density: np.ndarray,
        bulk_modulus: np.ndarray,
        f: np.ndarray
    ) -> np.ndarray:
    """Calculate the transfer matrix for a fluid layer.

    Parameters
    ----------
    thickness : float
        Thickness of the fluid layer.
    eff_density : np.ndarray
        Effective density of the fluid.
    bulk_modulus : np.ndarray
        Bulk modulus of the fluid.
    f : np.ndarray
        Frequencies.

    Returns
    -------
    np.ndarray
        Transfer matrix of the fluid layer.

    Notes
    -----
    The transfer matrix is calculated as follows:
    $$[T] = \begin{bmatrix}
    \cos(k_3h)&j\frac{\omega \rho}{k_3} \sin(k_3h)\\
    j\frac{k_3}{\omega \rho} \sin(k_3h) & \cos(k_3h)
    \end{bmatrix}$$

    where:
    - $k_3 = \omega \sqrt{\frac{\rho}{K}}$ is the normal component 
        of the wavenumber of the fluid layer,
    """
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

def tm_thin_plate(
        impedance: np.ndarray,
    ) -> np.ndarray:
    """Calculate the transfer matrix for a thin plate.

    Parameters
    ----------
    impedance : np.ndarray
        Mechanical impedance of the plate.

    Returns
    -------
    np.ndarray
        Transfer matrix of the thin plate.
    """
    t11 = np.ones_like(impedance)
    t12 = -impedance
    t21 = np.zeros_like(impedance)
    t22 = np.ones_like(impedance)
    return np.array([[t11, t12],[t21, t22]])

def tm_perforation(impedance: np.ndarray) -> np.ndarray:
    """Calculate the transfer matrix for a perforation.

    Parameters
    ----------
    impedance : np.ndarray
        Mechanical impedance of the perforation.

    Returns
    -------
    np.ndarray
        Transfer matrix of the perforation.
    """
    t11 = np.ones_like(impedance)
    t12 = impedance
    t21 = np.zeros_like(impedance)
    t22 = np.ones_like(impedance)
    return np.array([[t11, t12],[t21, t22]])
