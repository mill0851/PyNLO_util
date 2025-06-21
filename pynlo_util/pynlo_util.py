import numpy as np

c = 299792458.0 # m/s

def n_YAG(wl):
    """
    Returns the Sellmeier equation for YAG
    Args:
        wl: The wavelength to compute n at (um)
    returns:
        n: n computes at wl based on the Sellmeier equation (unitless)
    """
    # From Zelmon et al.
    B1, C1 = 2.282, 0.01185
    B2, C2 = 3.27644, 282.734

    wl2 = wl**2
    n = np.sqrt(1
                + B1*wl2/(wl2 - C1)
                + B2*wl2/(wl2 - C2)
                )
    return n

def wl_to_omega(wl):
    """Convert wavelength (m) to angular frequency (rad/s)"""
    return 2 * np.pi * c / wl

def omega_to_wl(omega):
    """Conver angular frequency (rad/s) to wavelength (m)"""
    return 2 * np.pi * c / omega

def omega_grid(N, center_wl, fractional_range):
    """
    Returns an angular frequency grid centered at the center wl. fractional range
    determines what fraction of the center wavelength +/- the range is.

    Args:
        N: number of grid points must be odd
        center_wl: center wavelength of grid
        fractional_range: fraction of the center wavelength +/-, determines grid bounds
    returns:
        omega_grid: omega grid centered at center_wl   
    """

    if (N % 2) == 0:
        raise ValueError("N must be odd")
    if fractional_range >= 1:
        raise ValueError("Fractional range must be less than 1")

    omega0 = wl_to_omega(center_wl)
    omega_range = omega0 * fractional_range
    omega_grid = omega0 + np.linspace(-omega_range, omega_range, N)
    return omega_grid

def propagation_const(omega, sellmeier_equation):
    """
    Computes the propagation constant beta(omega) for some sellmeier_equation at some wavelength

    Args:
        wl: center wavlength
        sellmeier_equation: sellmeier equation centered at wl
    returns:
        beta: returns the propagation cosntant as a function of angular freq
    """
    if type(omega) != np.ndarray and type(omega) != np.array:
        raise TypeError("Omega axis must be of type np.ndarray or np.array")
    if type(sellmeier_equation) != np.ndarray and type(sellmeier_equation) != np.array:
        raise TypeError("Sellmeier equation must be of type np.ndarray or np.array")
    return (sellmeier_equation * omega) / c

def dispersion_coeff(beta0, omega):
    """
    Calculates the second, third, and fourth dispersion coefficients. keep in mind your beta 0
    should have been made with the same omega axis that you pass to this function.

    Args:
        beta0: an array containing propagation constant values (1/m)
        omega: angular frequency axis (rad/s)
    returns:
        beta2, beta3, beta4: returns dispersion coefficients of order 2,3,4
        (s^2/m)  (s^3/m)  (s^4/m)
    
    """
    if len(omega) % 2 == 0:
        raise ValueError("Length of omega axis must be odd")
    if len(beta0) % 2 == 0:
        raise ValueError("Length of beta0 axis must be odd")
    if len(beta0) != len(omega):
        raise ValueError("beta0 and omega axis must be the same length")
    
    d1 = np.gradient(beta0, omega)
    d2 = np.gradient(d1, omega)
    d3 = np.gradient(d2, omega)
    d4 = np.gradient(d3, omega)

    N = len(omega)

    beta2 = d2[N//2]
    beta3 = d3[N//2]
    beta4 = d4[N//2]

    return beta2, beta3, beta4

