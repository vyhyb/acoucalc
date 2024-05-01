from typing import Tuple
import numpy as np

# material model for air
def air(
        f: np.ndarray,
        rho0: float=1.21, 
        c0: float=343.0
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the effective density and bulk modulus of air.

    This function calculates the effective density and bulk modulus of air based on the given frequency array `f`,
    reference density `rho0`, and speed of sound `c0`.

    Parameters
    ----------
    f : np.ndarray
        Array of frequencies.
    rho0 : float, optional
        Reference density of air, by default 1.21.
    c0 : float, optional
        Speed of sound in air, by default 343.0.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the effective density and bulk modulus of air.

    """
    z_c = rho0 * c0 * np.ones_like(f)
    
    eff_density = z_c / c0
    bulk_modulus = z_c**2 / eff_density
    return eff_density, bulk_modulus

### Delany-Bazley material model 
def delany(
        sigma: float, 
        f: np.ndarray, 
        rho0: float=1.21, 
        c0: float=343.0
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the effective density and bulk modulus of a fluid using the Delany-Bazley model.

    Parameters
    ----------
    sigma : float
        The static air flow resistivity of the fluid in Rayls/m^2.
    f : np.ndarray
        The frequency array in Hz.
    rho0 : float, optional
        The reference air density in kg/m^3. Default is 1.21 kg/m^3.
    c0 : float, optional
        The reference speed of sound in m/s. Default is 343.0 m/s.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the effective density and bulk modulus arrays.

    References
    ----------
    [1] M. E. Delany and E. N. Bazley, "Acoustical properties of fibrous absorbent materials," Applied Acoustics, vol. 3, no. 2, pp. 105-116, 1970
    """
    angular_f = 2*np.pi*f
    X = rho0*f/sigma
    wavenumber = angular_f/c0*(1 + 0.0978 * X**(-0.7) - 1j * 0.189 * X**(-0.595))
    z_c = rho0*c0*(1 + 0.0571 * X**(-0.754) - 1j * 0.087 * X**(-0.732))
    
    c = angular_f / wavenumber
    
    eff_density = z_c / c
    bulk_modulus = z_c**2 / eff_density
    return eff_density, bulk_modulus

### JCA material model
def jca_eff_density(
        tortuosity: float,
        flow_resistivity: float,
        porosity: float,
        viscous_char_length: float,
        angular_f: np.ndarray,
        rho_0 = 1.21,
        viscosity = 1.84e-5
        ) -> np.ndarray:
    """Calculate the effective density of a fluid using the Johnson-Champoux-Allard model.

    Parameters
    ----------
    angular_f : np.ndarray
        Angular frequency.
    tortuosity : float
        Tortuosity of the fluid.
    flow_resistivity : float
        Flow resistivity of the fluid.
    porosity : float
        Porosity of the fluid.
    viscous_char_length : float
        Viscous characteristic length of the fluid.
    rho_0 : float, optional
        Reference density of the fluid, by default 1.21.
    viscosity : float, optional
        Dynamic viscosity of the fluid, by default 1.84e-5.

    Returns
    -------
    np.ndarray
        The effective density of the fluid.

    References
    ----------
    [1] D. L. Johnson, J. Koplik, and R. Dashen, "Theory of dynamic permeability and tortuosity in fluid-saturated porous media", J. Fluid Mech. 176, 1987
    [2] T. J. Cox and P. D'Antonio, Acoustic Absorbers and Diffusers: Theory, Design and Application, 3rd ed. Taylor & Francis, 2017.
    """
    ed = (tortuosity * rho_0 / porosity * 
          (1 + flow_resistivity * porosity / (1j * angular_f * rho_0 * tortuosity) * np.sqrt(
              1 + (4j * tortuosity**2 * viscosity * rho_0 * angular_f)
              / (flow_resistivity**2 * viscous_char_length**2 * porosity**2)
          )))
    return ed

def jca_bulk_modulus(
        thermal_char_length: float,
        porosity: float,
        angular_f: np.ndarray,
        viscosity=1.84e-5,
        gamma=1.4,
        prandtl=0.77,
        rho_0=1.21,
        atm_pressure=101320
        ) -> np.ndarray:
    """Calculate the bulk modulus of a fluid using the 
    Johnson-Champoux-Allard (JCA) model.

    Parameters
    ----------
    angular_f : np.ndarray
        Array of angular frequencies.
    thermal_char_length : float
        Thermal characteristic length of the fluid.
    porosity : float
        Porosity of the fluid.
    viscosity : float, optional
        Dynamic viscosity of the fluid, by default 1.84e-5.
    gamma : float, optional
        Specific heat ratio of the fluid, by default 1.4.
    prandtl : float, optional
        Prandtl number of the fluid, by default 0.77.
    rho_0 : float, optional
        Reference density of the fluid, by default 1.21.
    atm_pressure : float, optional
        Atmospheric pressure, by default 101320.

    Returns
    -------
    np.ndarray
        Bulk modulus of the fluid.

    References
    ----------
    [1] Y. Champoux and J. Allard, "Dynamic tortuosity and bulk modulus in air-saturated porous media", J. Appl. Phys. 70, 1991
    [2] T. J. Cox and P. D'Antonio, Acoustic Absorbers and Diffusers: Theory, Design and Application, 3rd ed. Taylor & Francis, 2017.
    """
    bm = (gamma * atm_pressure / porosity /
          (gamma - (gamma-1)/(1 - (1j * 8 * viscosity)/
            (thermal_char_length**2 * prandtl * angular_f * rho_0)
            *np.sqrt(1 + 
                (1j * rho_0 * angular_f * prandtl * thermal_char_length**2)
                /(16*viscosity)))))
    return bm

def jca(
        flow_resistivity: float,
        porosity: float,
        tortuosity: float,
        viscous_char_length: float,
        thermal_char_length: float,
        f: np.ndarray,
        rho_air=1.21,
        viscosity=1.84e-5,
        gamma=1.4,
        prandtl=0.77,
        atm_pressure=101320
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the effective density and bulk modulus of a fluid
    using the Johnson-Champoux-Allard (JCA) model.

    Parameters
    ----------
    flow_resistivity : float
        Flow resistivity of the fluid.
    porosity : float
        Porosity of the fluid.
    tortuosity : float
        Tortuosity of the fluid.
    viscous_char_length : float
        Viscous characteristic length of the fluid.
    thermal_char_length : float
        Thermal characteristic length of the fluid.
    f : np.ndarray
        Array of frequencies.
    rho_air : float, optional
        Density of air. Default is 1.21 kg/m^3.
    viscosity : float, optional
        Viscosity of the fluid. Default is 1.84e-5 kg/(m*s).
    gamma : float, optional
        Specific heat ratio of the fluid. Default is 1.4.
    prandtl : float, optional
        Prandtl number of the fluid. Default is 0.77.
    atm_pressure : int, optional
        Atmospheric pressure. Default is 101320 Pa.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing the effective density and bulk modulus of the fluid.

    References
    ----------
    [1] D. L. Johnson, J. Koplik, and R. Dashen, "Theory of dynamic permeability and tortuosity in fluid-saturated porous media", J. Fluid Mech. 176, 1987
    [2] Y. Champoux and J. Allard, "Dynamic tortuosity and bulk modulus in air-saturated porous media", J. Appl. Phys. 70, 1991
    [3] T. J. Cox and P. D'Antonio, Acoustic Absorbers and Diffusers: Theory, Design and Application, 3rd ed. Taylor & Francis, 2017.
    """
    angular_f = 2*np.pi*f
    eff_density = jca_eff_density(
        tortuosity,
        flow_resistivity,
        porosity,
        viscous_char_length,
        angular_f,
        rho_air,
        viscosity
        )
    bulk_modulus = jca_bulk_modulus(
        thermal_char_length,
        porosity,
        angular_f,
        viscosity,
        gamma,
        prandtl,
        rho_air,
        atm_pressure
        )
    return eff_density, bulk_modulus