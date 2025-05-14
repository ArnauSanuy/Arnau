import numpy as np
import astropy.units as u
from astropy.constants import k_B, m_e, c, h
from scipy.special import kn
import matplotlib.pyplot as plt
from statistics import mean, pvariance, pstdev

# constant values
ALPHA = 0.3
BETA = 0.5
C_1 = 0.5
C_3 = 0.3
DELTA = 0.0005
F = 1
R_MIN = 3.0
R_MAX = 1e3
mec2 = m_e * c**2
F_total = []
alpha_b = 0.5

P6 = 1
T7 = 1
a = 1
v = np.logspace(8,21)


# P6 = np.linspace(0.8, 1.2, 50)
# T7 = np.linspace(0.5, 4, 50)


def T_from_theta(theta_e):
    return (theta_e * mec2 / k_B).to("K")


def theta_from_T(T_e):
    return (k_B * T_e / mec2).to_value("")


def x_m_appendix_B(m_dot):
    log_x_m = 3.6 + 1 / 4 * np.log10(m_dot)
    return 10**log_x_m


def g(theta_e):
    """Eq. (11) Mahadevan 1994.

    Parameters
    ----------
    theta_e : float
        dimensionless temperature
    """
    factor_1 = 1 / kn(2, 1 / theta_e)
    factor_2 = 2 + 2 * theta_e + 1 / theta_e
    factor_3 = np.exp(-(1 / theta_e))
    return factor_1 * factor_2 * factor_3


def alpha_c(
    theta_e,
    m_dot,
    alpha=ALPHA,
    c_1=C_1,
    r_min=R_MIN,
):
    """Eq. (34) Mahadevan 1994."""

    tau_es = (
        (23.87 * m_dot)
        * np.power(alpha / 0.3, -1)
        * np.power(c_1 / 0.5, -1)
        * np.power(r_min / 3, -1 / 2)
    )
    A = 1 + 4 * theta_e + 16 * np.power(theta_e, 2)

    result = -(np.log(tau_es) / np.log(A))

    return result


def vp(
    x_m,
    m_dot,
    m,
    theta_e,
    alpha=ALPHA,
    beta=BETA,
    c_1=C_1,
    c_3=C_3,
    r_min=R_MIN,
):
    """Eq. (24) Mahadevan 1994. It's the first equation of the two, but after
    having changed the values of s_1 or s_2, so it's a bit different from
    the paper"""

    T_e = T_from_theta(theta_e).value

    calc = (
        1.6898e-4
        * np.power(((1 - beta) * c_3 * m_dot) / (alpha * c_1 * m), 1 / 2)
        * np.power(T_e, 2)
        * np.power(r_min, -5 / 4)
        * x_m
    )
    return calc * u.Unit("s-1")


def F_theta(theta_e):
    """Eq. (28) Mahadevan 1994. To create the list of values for F, we make
    the two if, calling the index for each theta_e. If not, this would
    give an error."""

    return np.where(
        theta_e < 1,
        4 * ((2 * theta_e / np.pi**3) ** (1 / 2)) * (1 + 1.781 * (theta_e) ** (1.34))
        + 1.73
        * ((theta_e) ** (3 / 2))
        * (1 + 1.1 * theta_e + ((theta_e) ** 2) - 1.25 * (theta_e) ** (5 / 2)),
        (9 * theta_e / (2 * np.pi)) * (np.log(1.123 * theta_e + 0.48) + 1.5)
        + 2.3 * theta_e * (np.log(1.123 * theta_e) + 1.28),
    )


def one_minus_alpha_c(
    theta_e, x_m, m, m_dot, alpha=ALPHA, beta=BETA, c_1=C_1, c_3=C_3, r_min=R_MIN
):
    """Eq. (46) Mahadevan 1994. But to construct this equation we also use
    equations (47) and (48). It is important to notice that a value under 1
    for the factor divided by four, would give an error, as there is not the
    possibility to have log10(-x). Important to remember."""

    T_e = T_from_theta(theta_e).value

    factor_1 = (
        3.57e2
        * np.power(x_m / 1000, -3)
        * np.power(alpha / 0.3, -1 / 2)
        * (beta / 0.5)
        * np.power((1 - beta) / 0.5, -3 / 2)
        * np.power(c_1 / 0.5, -1 / 2)
        * np.power(c_3 / 0.3, -1 / 2)
        * np.power(r_min / 3, 3 / 4)
        * np.power(T_e / 1e9, -7)
        * g(theta_e)
        * np.power(m, 1 / 2)
        * np.power(m_dot, 1 / 2)
    )

    C_F = (
        1.46e3
        * np.power(x_m / 1000, -1)
        * np.power(alpha / 0.3, 1 / 2)
        * np.power((1 - beta) / 0.5, -1 / 2)
        * np.power(c_1 / 0.5, 1 / 2)
        * np.power(c_3 / 0.3, -1 / 2)
        * np.power(r_min / 3, 5 / 4)
        * np.power(T_e / 1e9, -1)
        * np.power(m, 1 / 2)
        * np.power(m_dot, -1 / 2)
    )

    return np.log10(factor_1 / 4 - 1) / np.log10(C_F)


def Q_e_plus(
    theta_e,
    m,
    m_dot,
    r_min=R_MIN,
    alpha=ALPHA,
    c_1=C_1,
    c_3=C_3,
    beta=BETA,
    delta=DELTA,
    f=F,
):
    """Eq. (14) Mahadevan 1994. Equation (11) will be used to obtain g(theta_e).
    This is the left hand side of the big equation we want to study."""

    addend_1 = (
        1.2e38
        * g(theta_e)
        * np.power(alpha, -2)
        * np.power(c_1, -2)
        * c_3
        * beta
        * m
        * np.power(m_dot, 2)
        * np.power(r_min, -1)
    )
    addend_2 = (
        delta * 9.39e38 * ((1 - beta) / f) * c_3 * m * m_dot * np.power(r_min, -1)
    )
    return (addend_1 + addend_2) * u.Unit("erg s-1")


def P_synch(
    theta_e,
    x_m,
    m,
    m_dot,
    r_min=R_MIN,
    alpha=ALPHA,
    c_1=C_1,
    c_3=C_3,
    beta=BETA,
):
    """Eq. (26) Mahadevan 1994. The first term of the right hand side, once
    we have changed all the values."""

    T_e = T_from_theta(theta_e).value
    value = (
        5.3e35
        * np.power(x_m / 1000, 3)
        * np.power(alpha / 0.3, -3 / 2)
        * np.power((1 - beta) / 0.5, 3 / 2)
        * np.power(c_1 / 0.5, -3 / 2)
        * np.power(c_3 / 0.3, 3 / 2)
        * np.power(r_min / 3, -7 / 4)
        * np.power(T_e / 1e9, 7)
        * np.power(m, 1 / 2)
        * np.power(m_dot, 3 / 2)
    )
    return value * u.Unit("erg s-1")


def P_compton(
    theta_e,
    x_m,
    m,
    m_dot,
    alpha_c,
    r_min=R_MIN,
    alpha=ALPHA,
    c_1=C_1,
    c_3=C_3,
    beta=BETA,
):
    """Eq. (35) Mahadevan 1994. Second term of the right hand side using
    the P_synch obtained before."""

    T_e = T_from_theta(theta_e).value
    mu_p = vp(x_m, m_dot, m, theta_e).value
    _P_synch = P_synch(theta_e, x_m, m, m_dot, r_min, alpha, c_1, c_3, beta)

    factor1 = _P_synch / (0.71 * (1 - alpha_c))
    factor2 = 6.2e19 * (T_e / 1e9)
    factor3 = mu_p

    return factor1 * (np.power(factor2 / factor3, 1 - alpha_c) - 1)


def P_bremms(
    theta_e,
    m,
    m_dot,
    alpha=ALPHA,
    c_1=C_1,
    r_max=R_MAX,
    r_min=R_MIN,
):
    """Eq. (29) Mahadevan 1994. The third term and using the definition F
    obtained before"""

    F_calc = F_theta(theta_e)
    calc = (
        4.78e34
        * np.power(alpha, -2)
        * np.power(c_1, -2)
        * np.log(r_max / r_min)
        * m
        * np.power(m_dot, 2)
        * F_calc
    )

    return calc * u.Unit("erg s-1")


def P_total(theta_e, x_m, m, m_dot, alpha_c):
    """This is just the sum of the three terms."""

    return (
        P_synch(theta_e, x_m, m, m_dot)
        + P_compton(theta_e, x_m, m, m_dot, alpha_c)
        + P_bremms(theta_e, m, m_dot)
    )


def compute_T_e_equilibirum(m, m_dot):
    """Solve numerically the equation for the equilibrium temperature.
    Equals the electron heating rate with the radiated power."""

    T_e = np.logspace(9, 10.5, 300) * u.K
    theta_e = theta_from_T(T_e)
    _x_m = x_m_appendix_B(m_dot)
    _alpha_c = alpha_c(theta_e, m_dot)

    Q_e_plus_values = Q_e_plus(theta_e, m, m_dot)
    P_total_values = P_total(theta_e, _x_m, m, m_dot, _alpha_c)

    equilibrium = np.abs(Q_e_plus_values - P_total_values)
    equilibrium_idx = np.argmin(equilibrium)

    return T_e[equilibrium_idx].value


def new_m_dot(m, P6, T7):
    calc = (
    4.16e-5
    *(10.6/5.17)
    *np.power(0.5, 2)
    *(m/1e8)
    *P6
    *np.power(T7, -5/2))
    
    return calc


def new_m_dot_1(m, T7, Lx, a, alpha_b):
    calc = (
    4.16e-5
    *np.power(alpha_b, 2)
    *np.power(T7, 1/4)
    *(m/1e8)
    *np.power(np.power(a*T7, 3), -1/2)
    *np.power(Lx/1e41, 1/2)
    )
    return calc

def L_synch(
    nu,
    m,
    m_dot,
    x_m,
    T,
    alpha=ALPHA,
    c_1=C_1,
    c_3=C_3,
    beta=BETA
):
    """Compute the synchrotron luminosity"""
    calc = (
    1.05e-24
    * np.power(1.6898e-4*np.sqrt(((1 - beta)*c_3)/(alpha*c_1))*x_m, 8/5)
    * np.power(m, 6/5)
    * np.power(m_dot, 4/5)
    * np.power(nu, 2/5)
    * np.power(T, 21/5)
    )
    
    return calc


def L_compton(
    nu,
    nu_p,
    alpha_c,
    m,
    m_dot,
    x_m,
    T,
    alpha=ALPHA,
    c_1=C_1,
    c_3=C_3,
    beta=BETA
):
    """Compute Inverse Compton luminosity"""
    _L_synch = L_synch(nu_p, m, m_dot, x_m, T, alpha, c_1, c_3, beta)

    calc = (
        np.power(nu_p, alpha_c)
        * _L_synch
        * np.power(nu, -alpha_c)
    )

    return calc


def L_bremms(
    nu,
    T_values,
    m,
    m_dot,
    alpha=ALPHA,
    c_1=C_1,
    r_min=R_MIN,
    r_max=R_MAX,
):
    """Compute Bremms luminosity"""
    F_calc = F_theta(theta_from_T(T_values * u.K))
    calc = (
    2.29e24
    * np.power(alpha, -2)
    * np.power(c_1, -2)
    * np.log(r_max / r_min)
    * F_calc
    * np.power(T_values, -1)
    * m
    * np.power(m_dot, 2)
    * np.exp(-((h*(nu * u.Unit("s-1")))/(k_B*(T_values * u.K))))
    )

    return calc.value


def Total_flux(m, m_dot, x_m, T_values, alpha_c, v, v1, v2, mu_p):
    """This is just the sum of the three terms."""

    first_part_flux = (L_synch(m, m_dot, x_m, T_values, v1) * v1)
    second_part_flux  = (L_compton(alpha_c, m, m_dot, x_m, T_values, v2, mu_p) 
                         + L_bremms(T_values, m, m_dot, v2)) * v2

    return first_part_flux.tolist(), second_part_flux.tolist()

def Fig_temperatures_1(m_dot, T_values, ref_m_dot, ref_T_e, ax=None):
    
    created_ax = False
    if ax is None:
        fig, ax = plt.subplots()
        created_ax = True
    
    
    ax.plot(ref_m_dot, ref_T_e, ls="--", label="reference")
    ax.plot(np.log10(m_dot), np.asarray(T_values) / 1e9, label="implementation")
    ax.legend()
    ax.set_ylim([0.8, 5.6])
    ax.set_xlim([-8, 0])
    ax.set_xlabel(r"$Log(\dot{m})$")
    ax.set_ylabel(r"$T_e\,/\,(10^{9}\,{\rm K})$")
    if created_ax:
        plt.show()
    
    return ax


def Fig_temperatures_2(m_dot, T_values, ref_m_dot, ref_T_e, ax=None):
    created_ax = False
    if ax is None:
        fig, ax = plt.subplots()
        created_ax = True
    

    ax.plot(ref_m_dot, ref_T_e, ls="--", label="reference")
    ax.plot(np.log10(m_dot), np.asarray(T_values) / 1e9, label="implementation")
    ax.legend()
    ax.set_ylim([0.8, 10])
    ax.set_xlim([-8, 0])
    ax.set_xlabel(r"$Log(\dot{m})$")
    ax.set_ylabel(r"$T_e\,/\,(10^{9}\,{\rm K})$")
    if created_ax:
        plt.show()
    
    return ax


def Fig_temperatures_3(m_dot, T_values, ref_m_dot, ref_T_e, ax=None):
    created_ax = False
    if ax is None:
        fig, ax = plt.subplots()
        created_ax = True

    ax.plot(ref_m_dot, ref_T_e, ls="--", label="reference")
    ax.plot(np.log10(m_dot), np.asarray(T_values) / 1e9, label="implementation")
    ax.legend()
    ax.set_ylim([0.8, 12])
    ax.set_xlim([-8, 0])
    ax.set_xlabel(r"$Log(\dot{m})$")
    ax.set_ylabel(r"$T_e\,/\,(10^{9}\,{\rm K})$")
    if created_ax:
        plt.show()
    
    return ax



def Figure_initial_flux(m, m_dot, ref_m_dot, ref_T_e, ax=None, color='b'):
    
    x_m = x_m_appendix_B(m_dot)
    T_calc = compute_T_e_equilibirum(m, m_dot)
    alpha = alpha_c(theta_from_T(T_calc * u.K), m_dot)
    mu_p = vp(x_m, m_dot, m, theta_from_T(T_calc * u.K)).value
    v_first = np.logspace(np.log10(mu_p), 8, 50)
    v_second = np.logspace(np.log10(mu_p),  21, 50)
    v_total = np.logspace(8, 21, 50)
    
    flux  = Total_flux(m, m_dot, x_m, T_calc, alpha, v_total, v_first, v_second, mu_p)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    ax.plot(ref_m_dot, ref_T_e, ls="--", color= color, label="reference")
    ax.plot(np.log10(v_first), np.log10(flux[0]), color = 'r', label = "implementation ") 
    ax.plot(np.log10(v_second), np.log10(flux[1]), color = 'r')
    ax.set_xlabel(r"$\log(\nu \, \mathrm{Hz})$")
    ax.set_ylabel(r"$\log(\nu L_{\nu} \, \mathrm{ergs} \, \mathrm{s}^{-1})$")
    ax.legend()

    return ax

def Figure_initial_flux_woref(m, m_dot, ax=None, color='b', label = 'label'):
    
    x_m = x_m_appendix_B(m_dot)
    T_calc = compute_T_e_equilibirum(m, m_dot)
    alpha = alpha_c(theta_from_T(T_calc * u.K), m_dot)
    mu_p = vp(x_m, m_dot, m, theta_from_T(T_calc * u.K)).value
    v_first = np.logspace(np.log10(mu_p), 8, 50)
    v_second = np.logspace(np.log10(mu_p),  21, 50)
    v_total = np.logspace(8, 21, 50)
    
    flux  = Total_flux(m, m_dot, x_m, T_calc, alpha, v_total, v_first, v_second, mu_p)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    ax.plot(np.log10(v_first), np.log10(flux[0]), color = color) 
    ax.plot(np.log10(v_second), np.log10(flux[1]), color = color, label=label)
    ax.set_xlabel(r"$\log(\nu \, \mathrm{Hz})$")
    ax.set_ylabel(r"$\log(\nu L_{\nu} \, \mathrm{ergs} \, \mathrm{s}^{-1})$")
    ax.legend()

    return ax


def final_flux(nu, m, m_dot):
    """Compute the total flux"""
    T_calc = compute_T_e_equilibirum(m, m_dot)

    new_x_m = x_m_appendix_B(m_dot)
    new_alpha_c = alpha_c(theta_from_T(T_calc * u.K), m_dot)
    mu_p = vp(new_x_m, m_dot, m, theta_from_T(T_calc * u.K)).value
    #mean_variable = mean(mu_p)

    v1 = np.logspace(np.log10(mu_p), 8, 50)
    v2 = np.logspace(np.log10(mu_p),  21, 50)

    flux = Total_flux(m, m_dot, new_x_m, T_calc, new_alpha_c, nu, v1, v2, mu_p)
    calculus = np.log10(flux)

    return v1, v2, calculus

def L_b(alpha_b, m, T, a, L_x):
    luminosity = (
        5.17e41
        * np.power(alpha_b, 2)
        * np.power(m, 2)
        * np.power(T, 1/4)
        * np.power(a*T, -3/2)
        * np.power(L_x/(1e41), 1/2)
    )
    return luminosity



