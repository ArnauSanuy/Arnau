import numpy as np
import astropy.units as u
from radiation import *
import matplotlib.pyplot as plt
from statistics import mean, pvariance, pstdev

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


# Test 3: try to reproduce Fig. 2 and Fig. 3 for the first mass
# let us make a larger and finer array of temperatures

data_m_10 = np.loadtxt("data/T_e_m_10.txt", delimiter=",")
ref_m_dot = data_m_10[:, 0]
ref_T_e = data_m_10[:, 1]

m = 10
# T_e = np.logspace(9, 10, 50) * u.K
# theta_e = theta_from_T(T_e)
m_dot = np.logspace(-8, 0, 50)
_x_m = x_m_appendix_B(m_dot)

T_eq_values = []

for _m_dot in m_dot:
    T_eq_values.append(compute_T_e_equilibirum(m, _m_dot))

T_final_values = T_eq_values * u.K

m_proof = 5e7
P6 = np.linspace(1.5, 3, 50)
T7 = np.linspace(0.2, 4, 50)

new_m_dot = (
    4.16e-5
    *(10.6/5.17)
    *np.power(32.2, 1/2)
    *np.power(1/0.3, 1/3)
    *np.power(4.3e-5, 1/6)
    *np.power(0.5, 8/3)
    *P6
    *np.power(T7, -5/2))

new_x_m = x_m_appendix_B(new_m_dot)
new_alpha_c = alpha_c(theta_from_T(T_final_values), new_m_dot)
mu_p = vp(new_x_m, new_m_dot, m_proof, theta_from_T(T_eq_values * u.K)).value
mean = mean(mu_p)

v1 = np.logspace(np.log10(mean), 8, 50)
v2 = np.logspace(np.log10(mean),  21, 50)
v = np.logspace(8, 21, 50)

first_part = v1*L_synch(m_proof, new_m_dot, new_x_m, T_eq_values, v1)
second_part = v2*L_compton(new_alpha_c, m_proof, new_m_dot, new_x_m, T_eq_values, v2, mean)
third_part = v2*L_bremms(T_eq_values, m_proof, new_m_dot, v2)
flux_2 = second_part + third_part
united_list = [item for sublist in [flux_2, first_part] for item in sublist]
v_list = [item for sublist in [v2, v1] for item in sublist]

first_flux = Total_flux(m_proof, new_m_dot, new_x_m, T_eq_values, new_alpha_c, v, v1, v2, mu_p)

calculus = np.log10(first_flux.value)

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

#plt.plot(np.log10(v_list), np.log10(united_list))
plt.plot(np.log10(v1), np.log10(first_part))
#plt.plot(np.log10(v2), np.log10(second_part))
#plt.plot(np.log10(v2), np.log10(third_part))
plt.plot(np.log10(v2), np.log10(second_part + third_part))
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

theta_e_final = theta_from_T(T_final_values)
alpha_first_mass = one_minus_alpha_c(theta_e_final, _x_m, m, m_dot)


# Test 3: try to reproduce Fig. 2 and Fig. 3 for the second mass
# let us make a larger and finer array of temperatures

data_m_1e5 = np.loadtxt("data/T_e_m_1e5.txt", delimiter=",")
ref_m_dot = data_m_1e5[:, 0]
ref_T_e = data_m_1e5[:, 1]


m = 1e5

T_eq_values_2 = []

for _m_dot in m_dot:
    T_eq_values_2.append(compute_T_e_equilibirum(m, _m_dot))

T_final_values_2 = T_eq_values_2 * u.K

fig, ax1 = plt.subplots()
ax1.plot(ref_m_dot, ref_T_e, ls="--", label="reference")
ax1.plot(np.log10(m_dot), np.asarray(T_eq_values_2) / 1e9, label="implementation")
# ax2 = ax1.twinx()
# ax2.semilogy(ref_m_dot_x_m, ref_x_m, ls="--", label="reference")
# ax2.semilogy(np.log10(m_dot), np.asarray(x_m_values_1), label="implementation")
ax1.legend()
ax1.set_ylim([0.8, 10])
ax1.set_xlim([-8, 0])
ax1.set_xlabel(r"$Log(\dot{m})$")
ax1.set_ylabel(r"$T_e\,/\,(10^{9}\,{\rm K})$")
plt.show()

# fig, ax = plt.subplots()
# ax.semilogy(np.log10(m_dot), np.asarray(x_m_values_2), label="implementation")
# ax.set_xlabel(r"$Log(\dot{m})$")
# ax.set_ylabel("x_m")
# # ax.set_xlim([0, -8])
# # ax.set_ylim([8e1, 2e3])
# plt.show()


theta_e_final = theta_from_T(T_final_values_2)
alpha_second_mass = one_minus_alpha_c(theta_e_final, _x_m, m, m_dot)

# Test 3: try to reproduce Fig. 2 and Fig. 3 for the third mass
# let us make a larger and finer array of temperatures

data_m_1e7 = np.loadtxt("data/T_e_1e7.txt", delimiter=",")
ref_m_dot = data_m_1e7[:, 0]
ref_T_e = data_m_1e7[:, 1]

m = 1e7

T_eq_values_3 = []

for _m_dot in m_dot:
    T_eq_values_3.append(compute_T_e_equilibirum(m, _m_dot))

T_final_values_3 = T_eq_values_3 * u.K


fig, ax1 = plt.subplots()
ax1.plot(ref_m_dot, ref_T_e, ls="--", label="reference")
ax1.plot(np.log10(m_dot), np.asarray(T_eq_values_3) / 1e9, label="implementation")
# ax2 = ax1.twinx()
# ax2.semilogy(ref_m_dot_x_m, ref_x_m, ls="--", label="reference")
# ax2.semilogy(np.log10(m_dot), np.asarray(x_m_values_1), label="implementation")
ax1.legend()
ax1.set_ylim([0.8, 10])
ax1.set_xlim([-8, 0])
ax1.set_xlabel(r"$Log(\dot{m})$")
ax1.set_ylabel(r"$T_e\,/\,(10^{9}\,{\rm K})$")
plt.show()



# fig, ax = plt.subplots()
# ax.semilogy(np.log10(m_dot), np.asarray(x_m_values_2), label="implementation")
# ax.set_xlabel(r"$Log(\dot{m})$")
# ax.set_ylabel("x_m")
# # ax.set_xlim([0, -8])
# # ax.set_ylim([8e1, 2e3])
# plt.show()

theta_e_final = theta_from_T(T_final_values_3)
alpha_third_mass = one_minus_alpha_c(theta_e_final, _x_m, m, m_dot)


# Test 3: try to reproduce Fig. 2 and Fig. 3 for the fourth mass
# let us make a larger and finer array of temperatures


data_m_1e9 = np.loadtxt("data/T_e_1e9.txt", delimiter=",")
ref_m_dot = data_m_1e9[:, 0]
ref_T_e = data_m_1e9[:, 1]

m = 1e9

T_eq_values_4 = []

for _m_dot in m_dot:
    T_eq_values_4.append(compute_T_e_equilibirum(m, _m_dot))

T_final_values_4 = T_eq_values_4 * u.K

fig, ax1 = plt.subplots()
ax1.plot(ref_m_dot, ref_T_e, ls="--", label="reference")
ax1.plot(np.log10(m_dot), np.asarray(T_eq_values_4) / 1e9, label="implementation")
# ax2 = ax1.twinx()
# ax2.semilogy(ref_m_dot_x_m, ref_x_m, ls="--", label="reference")
# ax2.semilogy(np.log10(m_dot), np.asarray(x_m_values_1), label="implementation")
ax1.legend()
ax1.set_ylim([0.8, 12])
ax1.set_xlim([-8, 0])
ax1.set_xlabel(r"$Log(\dot{m})$")
ax1.set_ylabel(r"$T_e\,/\,(10^{9}\,{\rm K})$")
plt.show()



# fig, ax = plt.subplots()
# ax.semilogy(np.log10(m_dot), np.asarray(x_m_values_2), label="implementation")
# ax.set_xlabel(r"$Log(\dot{m})$")
# ax.set_ylabel("x_m")
# # ax.set_xlim([0, -8])
# # ax.set_ylim([8e1, 2e3])
# plt.show()


theta_e_final = theta_from_T(T_final_values_4)
alpha_fourth_mass = one_minus_alpha_c(theta_e_final, _x_m, m, m_dot)


plt.plot(np.log10(m_dot), alpha_first_mass, label = 1)
plt.plot(np.log10(m_dot), alpha_second_mass, label = 2)
plt.plot(np.log10(m_dot), alpha_third_mass, label = 3)
plt.plot(np.log10(m_dot), alpha_fourth_mass, label = 4)
plt.xlabel(r"$Log(\dot{m})$")
plt.ylabel(r"$T_e\,/\,(10^{9}\,{\rm K})$")
plt.legend()
plt.show()
