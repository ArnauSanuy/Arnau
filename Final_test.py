import numpy as np
import astropy.units as u
from radiation import *
import matplotlib.pyplot as plt

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

ref_m_dot_x_m = [
    -1.531598846064255,
    -1.6934007059793226,
    -1.8752710136357935,
    -2.196428891397445,
    -2.5570965256499845,
    -3.0565500188141694,
    -3.0565500188141694,
    -3.615895285706607,
    -4.095405758927772,
    -4.575355229443279,
    -4.915954415954416,
    -5.236798724220107,
    -5.457739791073123,
    -5.698874733465927,
    -5.879741618735328,
    -6.020785178017882,
]

ref_x_m = [
    1364.0502461334681,
    1038.5866307928857,
    778.8933505501925,
    606.6748515313063,
    494.4988229572563,
    444.7599003416358,
    444.7599003416358,
    397.0063385704089,
    357.07364988578627,
    304.57787045619534,
    252.04884210522764,
    203.89446234923574,
    168.7297999352975,
    135.46402331036538,
    114.67690613367206,
    96.34727135733985,
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
T_e = np.linspace(1, 6, 100) * 1e9 * u.K
theta_e = theta_from_T(T_e)
m = 10
m_dot = 1e-4
x_m = x_m_appendix_B(m_dot)
_alpha_c = alpha_c(theta_e, m_dot)
_g = g(theta_e)
v_p = vp(x_m, m_dot, m, theta_e)
F_total = F_theta(theta_e)
# print(one_minus_alpha_c(theta_e, x_m, m, m_dot))
minus_alpha_c = one_minus_alpha_c(theta_e, x_m, m, m_dot)
logm_dot = np.logspace(-1.5, -8, 100)

# plt.plot(logm_dot, minus_alpha_c)
# plt.xlabel('log($\dot{m}$)')
# plt.ylabel('1-alphac')
# plt.title('Temperatures equilibri massa 4')
# plt.legend()
# plt.show()
# quit()

# Q_e_plus_values = Q_e_plus(theta_e, m, m_dot)
# P_synch_values = P_synch(theta_e, x_m, m, m_dot)
# #print(P_synch_values)
# P_compton_values = P_compton(P_synch_values, alpha_c, theta_e)
# #print(P_compton_values)
# P_bremms_values = P_bremms(theta_e)
# #print(P_bremms_values)
# P_total_values = P_total(theta_e, x_m, m, m_dot)
# #print(P_total_values)


# fig, ax = plt.subplots()
# ax.semilogy(T_e, Q_e_plus_values, label=r"$Q^{e+}$")
# ax.semilogy(T_e, P_synch_values, ls="--", label=r"$P_{\rm synch}$")
# ax.set_xlabel(r"$T_{e}$")
# ax.set_ylabel("Power / " + r"$({\rm erg}\,{\rm s}^{-1})$")
# ax.legend()
# plt.show()


# the powers are equals at T ~ 2.3 * 1e9 K
# but we only used synchrotron losses and the approximation in Appendix B for x_m


# Test 3: try to reproduce Fig. 2 and Fig. 3 for the first mass
# let us make a larger and finer array of temperatures

data_m_10 = np.loadtxt("data/T_e_m_10.txt", delimiter=",")
ref_m_dot = data_m_10[:, 0]
ref_T_e = data_m_10[:, 1]

m = 10
T_e = np.logspace(9, 10, 300) * u.K
theta_e = theta_from_T(T_e)
m_dot = np.logspace(-8, 0, 50)

T_eq_values = []

for _m_dot in m_dot:
    T_eq_values.append(compute_T_e_equilibirum(m, _m_dot))

print(T_eq_values)
fig, ax1 = plt.subplots()

ax1.plot(ref_m_dot, ref_T_e, ls="--", label="reference")
ax1.plot(np.log10(m_dot), np.asarray(T_eq_values) / 1e9, label="implementation")
# ax2 = ax1.twinx()
# ax2.semilogy(ref_m_dot_x_m, ref_x_m, ls="--", label="reference")
# ax2.semilogy(np.log10(m_dot), np.asarray(x_m_values_1), label="implementation")
ax1.legend()
ax1.set_ylim([0.8, 5.6])
ax1.set_xlim([-8, 0])
ax1.set_xlabel(r"$Log(\dot{m})$")
ax1.set_ylabel(r"$T_e\,/\,(10^{9}\,{\rm K})$")
plt.show()


"""
fig, ax = plt.subplots()
ax.semilogy(ref_m_dot2, ref_x_m, ls="--", label="reference")
ax.semilogy(np.log10(m_dot), np.asarray(x_m_values_1), label="implementation")
ax.set_xlabel(r"$Log(\dot{m})$")
ax.set_ylabel("x_m")
# ax.set_xlim([0, -8])
# ax.set_ylim([8e1, 2e3])
plt.show()


fig, ax1 = plt.subplots()
ax1.plot(np.log10(m_dot), np.asarray(T_eq_values_1) / 1e9, label="implementation")
ax2 = ax1.twinx()
ax2.semilogy(np.log10(m_dot), np.asarray(x_m_values_1), label="implementation")
ax1.set_xlabel(r"$Log(\dot{m})$")
ax1.set_ylabel(r"$T_e\,/\,(10^{9}\,{\rm K})$")
ax2.set_ylabel("x_m")
ax1.set_xlim([0, -8])
# ax.set_ylim([0, 6])
plt.show()

"""

_x_m = x_m_appendix_B(_m_dot)
alpha_first_mass = one_minus_alpha_c(theta_e, x_m, m, np.power(10, m_dot))

# Test 3: try to reproduce Fig. 2 and Fig. 3 for the second mass
# let us make a larger and finer array of temperatures

m = 1e5

T_eq_values_2 = []
x_m_values_2 = []
alpha_values_2 = []

# for _m_dot in m_dot:
#     info = alpha_c(theta_e, _m_dot)
#     alpha_values.append(info)

# print(type(alpha_values))

# array = np.array(alpha_values)
# alpha_c_list = array.tolist()
# print(type(alpha_c_list))

for _m_dot in m_dot:
    # fine loop on temperatures

    _x_m = x_m_appendix_B(_m_dot)
    v_p = vp(_x_m, _m_dot, m, theta_e)
    _alpha_c = alpha_c(theta_e, _m_dot)

    Q_e_plus_values = Q_e_plus(theta_e, m, _m_dot)
    P_synch_values = P_synch(theta_e, _x_m, m, _m_dot)
    P_compton_values = P_compton(P_synch_values, _alpha_c, theta_e, x_m, m_dot, m)
    P_bremms_values = P_bremms(theta_e)
    # P_total_values = P_total(theta_e, _x_m, m, _m_dot)
    equilibrium = np.abs(
        Q_e_plus_values - P_synch_values - P_compton_values - P_bremms_values
    )
    equilibrium_idx = np.argmin(equilibrium)
    T_eq_values_2.append(T_e[equilibrium_idx].value)
    x_m_values_2.append(x_m[equilibrium_idx])


fig, ax1 = plt.subplots()
ax1.plot(np.log10(m_dot), np.asarray(T_eq_values_2) / 1e9, label="implementation")
ax2 = ax1.twinx()
ax2.semilogy(np.log10(m_dot), np.asarray(x_m_values_2), label="implementation")
ax1.set_xlabel(r"$Log(\dot{m})$")
ax1.set_ylabel(r"$T_e\,/\,(10^{9}\,{\rm K})$")
ax2.set_ylabel("x_m")
ax1.set_xlim([0, -8])
# ax.set_ylim([0, 6])
plt.show()


# fig, ax = plt.subplots()
# ax.semilogy(np.log10(m_dot), np.asarray(x_m_values_2), label="implementation")
# ax.set_xlabel(r"$Log(\dot{m})$")
# ax.set_ylabel("x_m")
# # ax.set_xlim([0, -8])
# # ax.set_ylim([8e1, 2e3])
# plt.show()


new_x_m_2 = np.array(x_m_values_2)

alpha_second_mass = one_minus_alpha_c(theta_e, new_x_m_2, m, np.power(10, m_dot))


# Test 3: try to reproduce Fig. 2 and Fig. 3 for the third mass
# let us make a larger and finer array of temperatures


m = 1e7

T_eq_values_3 = []
x_m_values_3 = []
alpha_values_3 = []

# for _m_dot in m_dot:
#     info = alpha_c(theta_e, _m_dot)
#     alpha_values.append(info)

# print(type(alpha_values))

# array = np.array(alpha_values)
# alpha_c_list = array.tolist()
# print(type(alpha_c_list))

for _m_dot in m_dot:
    # fine loop on temperatures

    _x_m = x_m_appendix_B(_m_dot)
    v_p = vp(_x_m, _m_dot, m, theta_e)
    _alpha_c = alpha_c(theta_e, _m_dot)

    Q_e_plus_values = Q_e_plus(theta_e, m, _m_dot)
    P_synch_values = P_synch(theta_e, _x_m, m, _m_dot)
    P_compton_values = P_compton(P_synch_values, _alpha_c, theta_e, x_m, m_dot, m)
    P_bremms_values = P_bremms(theta_e)
    # P_total_values = P_total(theta_e, _x_m, m, _m_dot)
    equilibrium = np.abs(
        Q_e_plus_values - P_synch_values - P_compton_values - P_bremms_values
    )
    equilibrium_idx = np.argmin(equilibrium)
    T_eq_values_3.append(T_e[equilibrium_idx].value)
    x_m_values_3.append(x_m[equilibrium_idx])


fig, ax1 = plt.subplots()
ax1.plot(np.log10(m_dot), np.asarray(T_eq_values_3) / 1e9, label="implementation")
ax2 = ax1.twinx()
ax2.semilogy(np.log10(m_dot), np.asarray(x_m_values_3), label="implementation")
ax1.set_xlabel(r"$Log(\dot{m})$")
ax1.set_ylabel(r"$T_e\,/\,(10^{9}\,{\rm K})$")
ax2.set_ylabel("x_m")
ax1.set_xlim([0, -8])
# ax.set_ylim([0, 6])
plt.show()


# fig, ax = plt.subplots()
# ax.semilogy(np.log10(m_dot), np.asarray(x_m_values_3), label="implementation")
# ax.set_xlabel(r"$Log(\dot{m})$")
# ax.set_ylabel("x_m")
# # ax.set_xlim([0, -8])
# # ax.set_ylim([8e1, 2e3])
# plt.show()


new_x_m_3 = np.array(x_m_values_3)

alpha_third_mass = one_minus_alpha_c(theta_e, new_x_m_3, m, np.power(10, m_dot))


# Test 3: try to reproduce Fig. 2 and Fig. 3 for the fourth mass
# let us make a larger and finer array of temperatures


m = 1e9

T_eq_values_4 = []
x_m_values_4 = []
alpha_values_4 = []

# for _m_dot in m_dot:
#     info = alpha_c(theta_e, _m_dot)
#     alpha_values.append(info)

# print(type(alpha_values))

# array = np.array(alpha_values)
# alpha_c_list = array.tolist()
# print(type(alpha_c_list))

for _m_dot in m_dot:
    # fine loop on temperatures

    _x_m = x_m_appendix_B(_m_dot)
    v_p = vp(_x_m, _m_dot, m, theta_e)
    _alpha_c = alpha_c(theta_e, _m_dot)

    Q_e_plus_values = Q_e_plus(theta_e, m, _m_dot)
    P_synch_values = P_synch(theta_e, _x_m, m, _m_dot)
    P_compton_values = P_compton(P_synch_values, _alpha_c, theta_e, x_m, m_dot, m)
    P_bremms_values = P_bremms(theta_e)
    # P_total_values = P_total(theta_e, _x_m, m, _m_dot)
    equilibrium = np.abs(
        Q_e_plus_values - P_synch_values - P_compton_values - P_bremms_values
    )
    equilibrium_idx = np.argmin(equilibrium)
    T_eq_values_4.append(T_e[equilibrium_idx].value)
    x_m_values_4.append(x_m[equilibrium_idx])


fig, ax1 = plt.subplots()
ax1.plot(np.log10(m_dot), np.asarray(T_eq_values_4) / 1e9, label="implementation")
ax2 = ax1.twinx()
ax2.semilogy(np.log10(m_dot), np.asarray(x_m_values_4), label="implementation")
ax1.set_xlabel(r"$Log(\dot{m})$")
ax1.set_ylabel(r"$T_e\,/\,(10^{9}\,{\rm K})$")
ax2.set_ylabel("x_m")
ax1.set_xlim([0, -8])
# ax.set_ylim([0, 6])
plt.show()


# fig, ax = plt.subplots()
# ax.semilogy(np.log10(m_dot), np.asarray(x_m_values_4), label="implementation")
# ax.set_xlabel(r"$Log(\dot{m})$")
# ax.set_ylabel("x_m")
# # ax.set_xlim([0, -8])
# # ax.set_ylim([8e1, 2e3])
# plt.show()


new_x_m_4 = np.array(x_m_values_4)

alpha_fourth_mass = one_minus_alpha_c(theta_e, new_x_m_4, m, np.power(10, m_dot))

plt.plot(np.log10(m_dot), alpha_first_mass, label="$m=10$")
plt.plot(np.log10(m_dot), alpha_second_mass, label="$m=10^5$")
plt.plot(np.log10(m_dot), alpha_third_mass, label="$m=10^7$")
plt.plot(np.log10(m_dot), alpha_fourth_mass, label="$m=10^9$")
plt.xlabel("log($\dot{m}$)")
plt.ylabel("1-alphac")
plt.legend()
plt.title("Temperatures equilibri massa 4")
plt.show()


# print(alpha_first_mass)
# print(alpha_second_mass)
# print(alpha_third_mass)
# print(alpha_fourth_mass)
