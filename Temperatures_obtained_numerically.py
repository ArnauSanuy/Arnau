import numpy as np
import astropy.units as u
from radiation import *
import matplotlib.pyplot as plt
from statistics import mean, pvariance, pstdev


# Now, we are going to present the different plots of Fig. 2 from Rohan Mahadevan
# paper of 1997. The structure of x_m is based on his approximation formula.
# So, we have not taken into account the nummericall x_m.
# This are between the values -6.5 and -1.5

# The first mass we analyse is m=10, where there is the reference and the implementation

data_m_10 = np.loadtxt("data/T_e_m_10.txt", delimiter=",")
ref_m_dot_1 = data_m_10[:, 0]
ref_T_e_1 = data_m_10[:, 1]

m = 10
m_dot = np.logspace(-6.5, -1.5, 50)

T_eq_values = []

for _m_dot in m_dot:
    T_eq_values.append(compute_T_e_equilibirum(m, _m_dot))

T_final_values = T_eq_values * u.K

Figure_1 = Fig_temperatures_1(m_dot, T_eq_values, ref_m_dot_1, ref_T_e_1)

# ---------------------------------------------------------
# The second mass we analyse is m=1e5, where there is the reference and the implementation

data_m_1e5 = np.loadtxt("data/T_e_m_1e5.txt", delimiter=",")
ref_m_dot_2 = data_m_1e5[:, 0]
ref_T_e_2 = data_m_1e5[:, 1]

m = 1e5

T_eq_values_2 = []

for _m_dot in m_dot:
    T_eq_values_2.append(compute_T_e_equilibirum(m, _m_dot))

T_final_values_2 = T_eq_values_2 * u.K


Figure_2 = Fig_temperatures_2(m_dot, T_eq_values_2, ref_m_dot_2, ref_T_e_2)

# ---------------------------------------------
# The third mass we analyse is m=1e7, where there is the reference and the implementation

data_m_1e7 = np.loadtxt("data/T_e_1e7.txt", delimiter=",")
ref_m_dot_3 = data_m_1e7[:, 0]
ref_T_e_3 = data_m_1e7[:, 1]

m = 1e7

T_eq_values_3 = []

for _m_dot in m_dot:
    T_eq_values_3.append(compute_T_e_equilibirum(m, _m_dot))

T_final_values_3 = T_eq_values_3 * u.K


Figure_3 = Fig_temperatures_2(m_dot, T_eq_values_3, ref_m_dot_3, ref_T_e_3)

# --------------------------------------------
# The fourth mass we analyse is m=1e9, where there is the reference and the implementation

data_m_1e9 = np.loadtxt("data/T_e_1e9.txt", delimiter=",")
ref_m_dot_4 = data_m_1e9[:, 0]
ref_T_e_4 = data_m_1e9[:, 1]

m = 1e9

T_eq_values_4 = []

for _m_dot in m_dot:
    T_eq_values_4.append(compute_T_e_equilibirum(m, _m_dot))

T_final_values_4 = T_eq_values_4 * u.K

Figure_4 = Fig_temperatures_3(m_dot, T_eq_values_4, ref_m_dot_4, ref_T_e_4)

# ------------------------------------------------
# Now, we present the Fig. 3 of Rohan Mahadevan paper of 1997.
# The plots are similar to the ones presented in the paper, but the due to
# the structure of x_m there is some difference. However, the slope and and shape
# of the four lines is pretty similar.


fig, axs = plt.subplots(2, 2, figsize=(10, 8))
fig.subplots_adjust(hspace=0.3, wspace=0.3)  

ax1 = axs[0, 0]
Fig_temperatures_1(m_dot, T_eq_values, ref_m_dot_1, ref_T_e_1, ax=ax1)
ax2 = axs[0, 1]
Fig_temperatures_2(m_dot, T_eq_values_2, ref_m_dot_2, ref_T_e_2, ax=ax2)
ax3 = axs[1, 0]
Fig_temperatures_2(m_dot, T_eq_values_3, ref_m_dot_3, ref_T_e_3, ax=ax3)
ax4 = axs[1, 1]
Fig_temperatures_3(m_dot, T_eq_values_4, ref_m_dot_4, ref_T_e_4, ax=ax4)
plt.show()



one_minus_alpha_c_numerically_1 = 1 -alpha_c(theta_from_T(T_final_values), m_dot)
one_minus_alpha_c_numerically_2 = 1 -alpha_c(theta_from_T(T_final_values_2), m_dot)
one_minus_alpha_c_numerically_3 = 1 -alpha_c(theta_from_T(T_final_values_3), m_dot)
one_minus_alpha_c_numerically_4 = 1 -alpha_c(theta_from_T(T_final_values_4), m_dot)


plt.plot(np.log10(m_dot), one_minus_alpha_c_numerically_1, label="$m=10$")
plt.plot(np.log10(m_dot), one_minus_alpha_c_numerically_2, label="$m=10^{5}$")
plt.plot(np.log10(m_dot), one_minus_alpha_c_numerically_3, label="$m=10^{7}$")
plt.plot(np.log10(m_dot), one_minus_alpha_c_numerically_4, label="$m=10^{9}$")
plt.ylim(-2, 0.5)
plt.xlim(0,-8)
plt.xlabel(r"$log(\dot{m})$")
plt.ylabel(r"$1-\alpha_c$")
plt.legend()
plt.show()


