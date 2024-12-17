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
mec2 = m_e * c**2
F_total = []


def T_from_theta(theta_e):
    return (theta_e * mec2 / k_B).to("K")


def theta_from_T(T_e):
    return (k_B * T_e / mec2).to("")


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

    result = (-1) * np.log(tau_es) / np.log(A)

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
    T_e = T_from_theta(theta_e).value

    calc = (
        1.6898e14
        * np.power(((1 - beta) * c_3 * m_dot) / (alpha * c_1 * m), 1 / 2)
        * np.power(T_e / 1e9, 2)
        * np.power(r_min, -5 / 4)
        * x_m
    )
    return calc * u.Unit("s-1")


def F_theta(theta_e):
    for i in range(len(theta_e)):
        if theta_e[i] < 1:
            case_1 = 4 * ((2 * theta_e[i] / math.pi**3) ** (1 / 2)) * (
                1 + 1.781 * (theta_e[i]) ** (1.34)
            ) + 1.73 * ((theta_e[i]) ** (3 / 2)) * (
                1
                + 1.1 * (theta_e[i])
                + ((theta_e[i]) ** 2)
                - 1.25 * (theta_e[i]) ** (5 / 2)
            )
            F_total.append(case_1)
        if theta_e[i] > 1:
            case_2 = (9 * theta_e[i] / (2 * math.pi)) * (
                math.log(1.123 * theta_e[i] + 0.48) + 1.5
            ) + 2.3 * theta_e[i] * (math.log(1.123 * theta_e[i]) + 1.28)
            F_total.append(case_2)

    return F_total * u.Unit("")


# def one_minus_alpha_c(
#     theta_e,
#     x_m,
#     m,
#     m_dot,
# ):

#     T_e = T_from_theta(theta_e).value

#     factor_1 = np.log10((8.925e73*(g(theta_e)*np.power(m*m_dot, 1/2))/(np.power(x_m, 3)*np.power(T_e, 7)))-1)
#     factor_2 = np.log10((1.46e15*np.power(m/m_dot, 1/2))/(x_m*T_e))

#     return factor_1/factor_2


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
    addend_1 = (
        1.2e38
        * g
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
    P_synch,
    alpha_c,
    theta_e,
):
    T_e = T_from_theta(theta_e).value
    mu_p = vp(x_m, m_dot, m, theta_e).value

    factor1 = P_synch / (0.71 * (1 - alpha_c))
    factor2 = 6.2e19 * (T_e / 1e9)
    factor3 = mu_p

    return factor1 * (np.power(factor2 / factor3, 1 - alpha_c) - 1)


def P_bremms(
    theta_e,
    alpha=ALPHA,
    c_1=C_1,
    r_max=R_MAX,
    r_min=R_MIN,
):
    F_calc = F_total
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


def P_total(theta_e, x_m, m, m_dot):
    P_synchroton = P_synch(theta_e, x_m, m, m_dot).value

    Total_P = (
        P_synch(theta_e, x_m, m, m_dot).value
        + P_compton(P_synchroton, alpha_c, theta_e).value
        + P_bremms(theta_e).value
    )
    return Total_P * u.Unit("erg s-1")


## main code begins here

ref_theta_e = np.asarray(
    [
        0.1686,
        0.2530,
        0.3373,
        0.4216,
        0.5059,
        0.5902,
        0.6746,
        0.7589,
        0.8432,
        0.9275,
        1.0118,
        1.0961,
        1.1805,
        1.2648,
        1.3491,
        1.4334,
        1.5177,
        1.6021,
        1.6864,
    ]
)
ref_g_theta_e = np.asarray(
    [
        12.003,
        6.7292,
        4.5134,
        3.3386,
        2.6261,
        2.1540,
        1.8209,
        1.5746,
        1.3859,
        1.2369,
        1.1166,
        1.0175,
        0.9345,
        0.8640,
        0.8035,
        0.7509,
        0.7048,
        0.6641,
        0.6278,
    ]
)

# reference Log(m_dot) and temeperatures / 1e9, Figure 2
ref_m_dot = [
    -1.525325288880798,
    -1.5888487244168206,
    -1.65239527619576,
    -1.766473934991346,
    -1.8805294775440153,
    -2.0324956584806277,
    -2.2347859002476334,
    -2.4872615053875298,
    -2.802543942533021,
    -3.1427456895430788,
    -3.5458929660162335,
    -3.8986468329302446,
    -4.226342692522186,
    -4.566868066933082,
    -4.869875779089625,
    -5.122744360359111,
    -5.325219532069453,
    -5.489922762853354,
    -5.629475521343517,
    -5.731002060235151,
    -5.8325517153697035,
    -5.934332532933424,
    -6.023376300649857,
    -6.099729251004835,
]

ref_T_e = [
    1.011904865102275,
    1.1805956477893653,
    1.356625837602615,
    1.576600005201155,
    1.7892347656735352,
    2.0384972130479637,
    2.2656489740722443,
    2.4266536059894186,
    2.5288273996827297,
    2.5428820753762897,
    2.5421423556029445,
    2.541495100801267,
    2.5849305212423825,
    2.7017368967021795,
    2.906685506404644,
    3.192460059466535,
    3.478327077500094,
    3.7716028513885638,
    4.079603672015187,
    4.314279770108964,
    4.5562952753289006,
    4.871704851810435,
    5.143101101777928,
    5.385162839483698,
]


# Test 1: test g(theta_e) implementation using Table 1
# fig, ax = plt.subplots()
# ax.plot(
#     ref_theta_e,
#     ref_g_theta_e,
#     ls="",
#     marker="o",
#     label=r"$g(\theta_{e})$" + " reference",
# )
# ax.plot(
#     ref_theta_e,
#     g(ref_theta_e),
#     ls="-",
#     marker=".",
#     label=r"$g(\theta_{e})$" + " implementation",
# )
# ax.set_xlabel(r"$\theta_{e}$")
# ax.set_ylabel(r"$g(\theta_{e})$")
# ax.legend()
# plt.show()

# Test 2: test Q_e_plus and P_synchn equilibrium
# from Figure 2, it appears that for m = 10, m_dot = 10^{-2}
# the equilibrium temperature is around 4 * 1e9 K
T_e = np.linspace(1, 6, 1000) * 1e9 * u.K
theta_e = theta_from_T(T_e)
m = 10
m_dot = 1e-4
x_m = x_m_appendix_B(m_dot)
alpha_c = alpha_c(theta_e, m_dot)
g = g(theta_e)
v_p = vp(x_m, m_dot, m, theta_e)
F_total = F_theta(theta_e)
# print(one_minus_alpha_c(theta_e, x_m, m, m_dot))
# minus_alpha_c = one_minus_alpha_c(x_m, m, m_dot)
logm_dot = np.logspace(-1.5, -8, 1000)


# # plt.plot(logm_dot, minus_alpha_c)
# # plt.xlabel('log($\dot{m}$)')
# # plt.ylabel('1-alphac')
# # plt.title('Temperatures equilibri massa 4')
# # plt.legend()
# # plt.show()


Q_e_plus_values = Q_e_plus(theta_e, m, m_dot)
P_synch_values = P_synch(theta_e, x_m, m, m_dot)
# print(P_synch_values)
P_compton_values = P_compton(P_synch_values, alpha_c, theta_e)
# print(P_compton_values)
P_bremms_values = P_bremms(theta_e)
# print(P_bremms_values)
P_total_values = P_total(theta_e, x_m, m, m_dot)
# print(P_total_values)


# fig, ax = plt.subplots()
# ax.semilogy(T_e, Q_e_plus_values, label=r"$Q^{e+}$")
# ax.semilogy(T_e, P_synch_values, ls="--", label=r"$P_{\rm synch}$")
# ax.set_xlabel(r"$T_{e}$")
# ax.set_ylabel("Power / " + r"$({\rm erg}\,{\rm s}^{-1})$")
# ax.legend()
# plt.show()


# the powers are equals at T ~ 2.3 * 1e9 K
# but we only used synchrotron losses and the approximation in Appendix B for x_m

# Test 3: try to reproduce Fig. 2
# let us make a larger and finer array of temperatures
T_e = np.logspace(8, 10, 1000) * u.K
theta_e = theta_from_T(T_e)
m_dot_values = np.logspace(-8, 0, 40)

# F_total = F_theta(theta_e)
# m = 10
# g = g(theta_e)
# alpha_c = alpha_c(theta_e, m_dot_values)


T_eq_values = []

for _m_dot in m_dot_values:
    print(f"solving equilibrium equation for m_dot = {_m_dot:.2e}")
    # fine loop on temperatures
    _x_m = x_m_appendix_B(_m_dot)

    # v_p = vp(_x_m, _m_dot, m, theta_e)
    # _alpha_c = alpha_c(theta_e, _m_dot)

    Q_e_plus_values = Q_e_plus(theta_e, m, _m_dot)
    P_synch_values = P_synch(theta_e, _x_m, m, _m_dot)
    P_compton_values = P_compton(P_synch_values, alpha_c, theta_e)
    P_bremms_values = P_bremms(theta_e)
    P_total_values = P_total(theta_e, _x_m, m, _m_dot)
    equilibrium = np.abs(Q_e_plus_values - P_total_values)
    equilibrium_idx = np.argmin(equilibrium)
    print(f"-> equilibrium temperature: {T_e[equilibrium_idx]:.2e}")
    T_eq_values.append(T_e[equilibrium_idx].value)


fig, ax = plt.subplots()
ax.plot(ref_m_dot, ref_T_e, ls="--", label="reference")
ax.plot(np.log10(m_dot_values), np.asarray(T_eq_values) / 1e9, label="implementation")
ax.set_xlabel(r"$Log(\dot{m})$")
ax.set_ylabel(r"$T_e\,/\,(10^{9}\,{\rm K})$")
ax.set_xlim([0, -8])
ax.set_ylim([1, 6])
plt.show()
