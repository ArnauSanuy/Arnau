import numpy as np
import astropy.units as u
from astropy.constants import k_B, m_e, c
from scipy.special import kn
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import math

# constant values
ALPHA = 0.3
BETA = 0.5
C_1 = 0.5
C_3 = 0.3
DELTA = 0.0005
F = 1
R_MIN = 3.0
R_MAX = 1e3
B = 1
E = 4.8032 * 10 ** (-10)
mec2 = m_e * c**2
F_total = []


Ac1 = np.linspace(0.94, 1.4, 51)
Ac2 = np.linspace(0.96, 1.3, 51)
vb = (E * B) / (2 * math.pi * m_e * c)
kTe = np.linspace(5e4, 1e6, 51)
theta_e = (kTe / mec2).value
v = np.logspace(12, 21, 51)
logv = np.linspace(12, 21, 51)

m = [10, 1e5, 1e7, 1e9]
m_dot_1 = np.logspace(-5 + math.log10(9), -7, 51)
m_dot_2 = np.logspace(-4 + math.log10(3), -4, 51)
m_dot_3 = np.logspace(-4, -3 + math.log10(8), 51)
alpha_c_1 = np.linspace(1.01, 10, 51)
alpha_c_2 = np.linspace(0.50, 1, 51)
alpha_c_3 = np.linspace(1.01, 10, 51)

# tau1 = (23.87*mp1)*(alpha/0.3)**(-1)*(c1/0.5)**(-1)*(rmin/3)**(-1/2)
# tau2 = (23.87*mp2)*((alpha/0.3)**(-1))*((c1/0.5)**(-1))*((rmin/3)**(-1/2))
# tau3 = (23.87*mp3)*(alpha/0.3)**(-1)*(c1/0.5)**(-1)*(rmin/3)**(-1/2)

# -----------------------------------------------
# -----------------------------------------------
# Here, appear all the definitions we will use or that are important


def T_from_theta(theta_e):
    return (theta_e * mec2 / k_B).to("K")


def theta_from_T(T_e):
    return (k_B * T_e / mec2).to("")


def x_m_appendix_B(m_dot):
    log_x_m = 3.6 + 1 / 4 * np.log10(m_dot)
    return 10**log_x_m


def tau_es(m_dot, alpha=ALPHA, c_1=C_1, r_min=R_MIN):
    factor = (
        23.87
        * m_dot
        * np.power(alpha / 0.3, -1)
        * np.power(c_1 / 0.5, -1)
        * np.power(r_min / 3, -1 / 2)
    )

    return factor


def alpha_c(
    theta_e,
    m_dot,
    tau_es,
    alpha=ALPHA,
    c_1=C_1,
    r_min=R_MIN,
):
    """Eq. (34) Mahadevan 1994."""

    A = 1 + 4 * theta_e + 16 * np.power(theta_e, 2)

    result = (-1) * np.log(tau_es) / np.log(A)

    return result


def temperature_1(
    x_m,
    m,
    m_dot,
    Ac1,
    alpha=ALPHA,
    beta=BETA,
    delta=DELTA,
    c_1=C_1,
    c_3=C_3,
    r_min=R_MIN,
):
    Temperature = (
        (1.1e9 / np.power(Ac1, 1 / 7))
        * np.power(2000 * delta, 1 / 7)
        * np.power(x_m / 300, -3 / 7)
        * np.power(alpha / 0.3, 3 / 14)
        * np.power((1 - beta) / 0.5, -1 / 14)
        * np.power(c_1 / 0.5, 3 / 14)
        * np.power(c_3 / 0.3, -1 / 14)
        * np.power(r_min / 3, 3 / 28)
        * np.power(m, 1 / 14)
        * np.power(m_dot, -1 / 14)
    )

    return Temperature


def temperature_2(tau_es, alpha_c):
    factor = 0.744e9 * (np.power((4 * np.power(tau_es, -1 / alpha_c) - 3), 1 / 2) - 1)

    return factor


def tempearature_3(
    x_m,
    m,
    m_dot,
    Ac2,
    alpha=ALPHA,
    beta=BETA,
    delta=DELTA,
    c_1=C_1,
    c_3=C_3,
    r_min=R_MIN,
):
    Temperature = (
        (2.7e9 / np.power(Ac2, 3 / 25))
        * np.power(x_m / 1000, -2 / 5)
        * np.power(alpha / 0.3, -3 / 50)
        * np.power(beta / 0.5, 3 / 25)
        * np.power((1 - beta) / 0.5, -1 / 5)
        * np.power(c_1 / 0.5, -3 / 50)
        * np.power(c_3 / 0.3, -3 / 50)
        * np.power(r_min / 3, 1 / 10)
        * np.power(m, 3 / 50)
        * np.power(m_dot, 3 / 50)
    )

    return Temperature


# -----------------------------------------
# -----------------------------------------
# Now, we call all the definitions to make the different temperatures
# for each mass


x_m_1 = x_m_appendix_B(m_dot_1)
x_m_2 = x_m_appendix_B(m_dot_2)
x_m_3 = x_m_appendix_B(m_dot_3)

temperature_1_first_mass = temperature_1(x_m_1, m[0], m_dot_1, Ac1)
# print(temperature_1_first_mass)
temperature_1_second_mass = temperature_1(x_m_1, m[1], m_dot_1, Ac1)
temperature_1_third_mass = temperature_1(x_m_1, m[2], m_dot_1, Ac1)
temperature_1_fourth_mass = temperature_1(x_m_1, m[3], m_dot_1, Ac1)


tau = tau_es(m_dot_2)
alphac = alpha_c(theta_e, m_dot_2, tau)

temperature_2 = temperature_2(tau, alpha_c_2)


temperature_3_first_mass = tempearature_3(x_m_3, m[0], m_dot_3, Ac2)
temperature_3_second_mass = tempearature_3(x_m_3, m[1], m_dot_3, Ac2)
temperature_3_third_mass = tempearature_3(x_m_3, m[2], m_dot_3, Ac2)
temperature_3_fourth_mass = tempearature_3(x_m_3, m[3], m_dot_3, Ac2)

# ---------------------------------------------
# ---------------------------------------------
# Now, there is the code to make the three plots


plt.plot(np.log10(m_dot_1), temperature_1_first_mass, label="m=10")
plt.plot(np.log10(m_dot_1), temperature_1_second_mass, label="$m=10^5$")
plt.plot(np.log10(m_dot_1), temperature_1_third_mass, label="$m=10^7$")
plt.plot(np.log10(m_dot_1), temperature_1_fourth_mass, label="$m=10^9$")
plt.xlabel("log(m_dot)")
plt.ylabel("Te (K)")
plt.title("Temperatures relations for every mass")
plt.legend()
plt.show()


plt.semilogy(np.log10(m_dot_2), temperature_2)
plt.xlabel("log(m_dot)")
plt.ylabel("Te (K)")
plt.title("Temperatures relation")
plt.show()


plt.plot(np.log10(m_dot_3), temperature_3_first_mass, label="m=10")
plt.plot(np.log10(m_dot_3), temperature_3_second_mass, label="$m=10^5$")
plt.plot(np.log10(m_dot_3), temperature_3_third_mass, label="$m=10^7$")
plt.plot(np.log10(m_dot_3), temperature_3_fourth_mass, label="$m=10^9$")
plt.xlabel("log(m_dot)")
plt.ylabel("Te (K)")
plt.title("Temperatures relations for every mass")
plt.legend()
plt.show()
