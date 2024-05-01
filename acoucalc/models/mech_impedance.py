"""Module containing different simple mechanical impedance models.
"""
import numpy as np

def thin_plate(
        thickness: float,
        density: float, 
        youngs_modulus: float, 
        speed_of_sound: float, 
        f: np.ndarray, 
        incidence_angle=0, 
        rm=0
        ) -> np.ndarray:
    """Calculates the mechanical impedance of a thin plate.

    This function calculates the mechanical impedance of a thin plate using
    the given parameters.

    Parameters
    ----------
    thickness : float
        The thickness of the thin plate in meters.
    density : float
        The density of the material of the thin plate in kg/m^3.
    youngs_modulus : float
        The Young's modulus of the material of the thin plate in Pa.
    speed_of_sound : float
        The speed of sound in the thin plate in m/s.
    f : np.ndarray
        The frequency values at which to calculate
        the mechanical impedance in Hz.
    incidence_angle : int, optional
        The incidence angle of the sound wave on the thin plate
        in degrees, by default 0.
    rm : int, optional
        The added resistance of the thin plate, by default 0.
        It is advised to use complex Young's modulus to include damping.

    Returns
    -------
    np.ndarray
        The mechanical impedance of the thin plate
        at the given frequency values.

    References
    ----------
    [1] J. F. Allard and N. Atalla, Propagation of Sound in Porous Media: Modelling Sound Absorbing Materials, 2nd ed. Wiley, 2009.
    """
    angular_f = 2*np.pi*f
    mass_per_area = density*thickness
    bending_stiffness = 1/12 * youngs_modulus*thickness**3
    w_c = speed_of_sound**2 * np.sqrt(mass_per_area/bending_stiffness)
    impedance = rm + 1j*angular_f*mass_per_area*(
        1 - (angular_f/w_c)**2 * np.sin(incidence_angle)**4
    )
    return impedance

def simple_circ_perforation(
        thickness: float, 
        hole_radius: float, 
        hole_spacing: float, 
        f: np.ndarray, 
        rho=1.21
        ) -> np.ndarray:
    """Calculate the mechanical impedance of a simple circular perforation.

    This function calculates the mechanical impedance of a simple circular perforation
    based on the given parameters. No damping is considered.

    Parameters
    ----------
    thickness : float
        The thickness of the material (length of the neck).
    hole_radius : float
        The radius of the hole.
    hole_spacing : float
        The spacing between the holes.
    f : np.ndarray
        The frequency array.
    rho : float, optional
        The density of the medium, defaults to 1.21.

    Returns
    -------
    np.ndarray
        The mechanical impedance array.

    References
    ----------
    [1] T. J. Cox and P. D'Antonio, Acoustic Absorbers and Diffusers: Theory, Design and Application, 3rd ed. Taylor & Francis, 2017.
    """
    angular_f = 2*np.pi*f
    hole_area = np.pi*hole_radius**2
    hole_spacing_area = hole_spacing**2
    porosity = hole_area/hole_spacing_area

    delta = 0.8*(1 - 1.47*porosity**0.5 + 0.47*porosity**1.5)
    m = rho/porosity * (thickness + 2*delta*hole_radius)

    impedance = 1j*angular_f*m
    return impedance

def viscous_damped_circ_perforation(
        thickness: float, 
        hole_radius: float, 
        hole_spacing: float, 
        f: np.ndarray, 
        method="guess75",
        rho=1.21,  
        kin_viscosity=15e-6,
        dyn_viscosity=1.84e-5
        ) -> np.ndarray:
    """
    Calculate the viscous damped mechanical impedance of a circular perforation.

    Parameters
    ----------
    thickness : float
        The thickness of the material (length of the neck).
    hole_radius : float
        The radius of the hole.
    hole_spacing : float
        The spacing between the holes.
    f : np.ndarray
        The frequency array.
    method : str, optional
        Method for calculating the impedance. Valid options are "guess75" and "ingard53". 
        Default is "guess75".
    rho : float, optional
        Density of the fluid. Default is 1.21 kg/m^3.
    kin_viscosity : float, optional
        Kinematic viscosity of the fluid. Default is 15e-6 m^2/s.
    dyn_viscosity : float, optional
        Dynamic viscosity of the fluid. Default is 1.84e-5 kg/(m*s).

    Returns
    -------
    np.ndarray
        Array of complex mechanical impedances corresponding to the input frequencies.

    References
    ----------
    [1] T. J. Cox and P. D'Antonio, Acoustic Absorbers and Diffusers: Theory, Design and Application, 3rd ed. Taylor & Francis, 2017.
    [2] A. W. Guess, "Calculation of perforated plate liner parameters from specified acoustic resistance and reactance", J. Sound Vib., 1975.
    [3] U. Ingard, "On the theory and design of acoustic resonators", J. Acoust. Soc. Am., 1953.
    [4] S. N. Rschevkin, Gestaltung von Resonanzschallschallschluckern und deren Verwendung fur die Nachhallregelung und Schallabsorption, Hochfrequenztechnik und Electrokustik, 1959
    [5] L. Cremer and H. A. Müller, Principles and Applications of Room Acoustics, Elsevier, 1984
    """
    angular_f = 2*np.pi*f
    hole_area = np.pi*hole_radius**2
    hole_spacing_area = hole_spacing**2
    porosity = hole_area/hole_spacing_area
    
    if method.lower() == "guess75":
        rm = rho / porosity * np.sqrt(8*kin_viscosity*angular_f)*(
            1 + thickness/(2*hole_radius)
        )
    elif method.lower() == "ingard53":
        rm = np.sqrt(2*rho*dyn_viscosity*angular_f)/(2*porosity)
    
    # Rschevkin's correction [4] found in [5]
    delta = 0.8*(1 - 1.47*porosity**0.5 + 0.47*porosity**1.5) 
    
    m = rho/porosity * (
        thickness + 2*delta*hole_radius + np.sqrt(
            8*kin_viscosity/angular_f * (1 + thickness/(2*hole_radius))
        )
    )

    impedance = rm + 1j*angular_f*m
    return impedance
