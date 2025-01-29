# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 20:49:52 2025

@author: HP
"""

import numpy as np
import astropy.units as u
from radiation import *
import matplotlib.pyplot as plt
from statistics import mean, pvariance, pstdev

# Now, we are going to present the different plots of Fig. 1 from Rohan Mahadevan
# paper of 1997. The structure of x_m is based on his approximation formula.
# So, we have not taken into account the nummerical x_m. Here, we won't change
# the mass, it will be changed the mass ratio (m_dot)

# The first mass ratio is m_dot=3e-4, where there is the reference and the implementation

data_Flux_m_3e4 = np.loadtxt("data/Flux_m_3e-4.txt", delimiter=",")
ref_m_dot_1 = data_Flux_m_3e4[:, 0]
ref_T_e_1 = data_Flux_m_3e4[:, 1]


m = 5e9
m_dot_1 = 3e-4
# x_m_1 = x_m_appendix_B(m_dot_1)
# T_calc_1 = compute_T_e_equilibirum(m, m_dot_1)
# alpha_1 = alpha_c(theta_from_T(T_calc_1 * u.K), m_dot_1)
# mu_p_1 = vp(x_m_1, m_dot_1, m, theta_from_T(T_calc_1 * u.K)).value
# v_first = np.logspace(np.log10(mu_p_1), 8, 50)
# v_second = np.logspace(np.log10(mu_p_1),  21, 50)
# v_total = np.logspace(8, 21, 50)


#proof  = Total_flux(m, m_dot_1, x_m_1, T_calc_1, alpha_1, v_total, v_first, v_second, mu_p_1)
Figure_initial_flux(m, m_dot_1, ref_m_dot_1, ref_T_e_1, ax=None)


# The second mass ratio is m_dot=6e-4, where there is the reference and the implementation

data_Flux_m_6e4 = np.loadtxt("data/Flux_m_6e-4.txt", delimiter=",")
ref_m_dot_2 = data_Flux_m_6e4[:, 0]
ref_T_e_2 = data_Flux_m_6e4[:, 1]


m_dot_2 = 6e-4
# x_m_2 = x_m_appendix_B(m_dot_2)
# T_calc_2 = compute_T_e_equilibirum(m, m_dot_2)
# alpha_2 = alpha_c(theta_from_T(T_calc_2 * u.K), m_dot_2)
# mu_p_2 = vp(x_m_2, m_dot_2, m, theta_from_T(T_calc_2 * u.K)).value
# v_first_2 = np.logspace(np.log10(mu_p_2), 8, 50)
# v_second_2 = np.logspace(np.log10(mu_p_2),  21, 50)


# flux_2  = Total_flux(m, m_dot_2, x_m_2, T_calc_2, alpha_2, v_total, v_first_2, v_second_2, mu_p_2)
Figure_initial_flux(m, m_dot_2, ref_m_dot_2, ref_T_e_2, ax=None)


# # The third mass ratio is m_dot=12e-4, where there is the reference and the implementation

data_Flux_m_12e4 = np.loadtxt("data/Flux_m_12e-4.txt", delimiter=",")
ref_m_dot_3 = data_Flux_m_12e4[:, 0]
ref_T_e_3 = data_Flux_m_12e4[:, 1]


m_dot_3 = 12e-4
# x_m_3 = x_m_appendix_B(m_dot_3)
# T_calc_3 = compute_T_e_equilibirum(m, m_dot_3)
# alpha_3 = alpha_c(theta_from_T(T_calc_3 * u.K), m_dot_3)
# mu_p_3 = vp(x_m_3, m_dot_3, m, theta_from_T(T_calc_3 * u.K)).value
# v_first_3 = np.logspace(np.log10(mu_p_3), 8, 50)
# v_second_3 = np.logspace(np.log10(mu_p_3),  21, 50)


# flux_3  = Total_flux(m, m_dot_3, x_m_3, T_calc_3, alpha_3, v_total, v_first_3, v_second_3, mu_p_3)
Figure_initial_flux(m, m_dot_3, ref_m_dot_3, ref_T_e_3, ax=None)

# # The fourth mass ratio is m_dot=24e-4, where there is the reference and the implementation

data_Flux_m_24e4 = np.loadtxt("data/Flux_m_24e-4.txt", delimiter=",")
ref_m_dot_4 = data_Flux_m_24e4[:, 0]
ref_T_e_4 = data_Flux_m_24e4[:, 1]


m_dot_4 = 24e-4
# x_m_4 = x_m_appendix_B(m_dot_4)
# T_calc_4 = compute_T_e_equilibirum(m, m_dot_4)
# alpha_4 = alpha_c(theta_from_T(T_calc_4 * u.K), m_dot_4)
# mu_p_4 = vp(x_m_4, m_dot_4, m, theta_from_T(T_calc_4 * u.K)).value
# v_first_4 = np.logspace(np.log10(mu_p_4), 8, 50)
# v_second_4 = np.logspace(np.log10(mu_p_4),  21, 50)

# flux_4  = Total_flux(m, m_dot_4, x_m_4, T_calc_4, alpha_4, v_total, v_first_4, v_second_4, mu_p_4)
Figure_initial_flux(m, m_dot_4, ref_m_dot_4, ref_T_e_4, ax=None)


# # Now,what we do is to present the four implementations in just on plot.


m_dot = [3e-4, 6e-4, 12e-4, 24e-4]

fig, ax = plt.subplots(figsize=(10, 6))
Figure_initial_flux(m, m_dot_1, ref_m_dot_1, ref_T_e_1, ax=ax, color = 'magenta')
Figure_initial_flux(m, m_dot_2, ref_m_dot_2, ref_T_e_2, ax=ax, color = 'purple')
Figure_initial_flux(m, m_dot_3, ref_m_dot_3, ref_T_e_3, ax=ax, color = 'green')
Figure_initial_flux(m, m_dot_4, ref_m_dot_4, ref_T_e_4, ax=ax, color = 'orange')
#ax.legend(['m_dot=3e-4', 'm_dot=6e-4', 'm_dot=12e-4', 'm_dot=24e-4'])
plt.show()


# plt.plot(np.log10(v_first), np.log10(proof[0]), color = 'b', label="$m=3 x 10^{-4}$") 
# plt.plot(np.log10(v_second), np.log10(proof[1]), color = 'b')
# plt.plot(np.log10(v_first_2), np.log10(flux_2[0]), color = 'orange', label="$m=6x10^{-4}$") 
# plt.plot(np.log10(v_second_2), np.log10(flux_2[1]), color = 'orange')
# plt.plot(np.log10(v_first_3), np.log10(flux_3[0]), color = 'green', label="$m=12x10^{-4}$") 
# plt.plot(np.log10(v_second_3), np.log10(flux_3[1]), color = 'green')
# plt.plot(np.log10(v_first_4), np.log10(flux_4[0]), color = 'red', label="$m=24x10^{-4}$") 
# plt.plot(np.log10(v_second_4), np.log10(flux_4[1]), color = 'red')
# plt.xlabel(r"$log(\nu \hspace{0.7} Hz)$")
# plt.ylabel(r"$\log(\nu L_{\nu} \hspace{1} ergs \hspace{0.2} s^{-1})$")
# plt.legend()
# plt.show()



