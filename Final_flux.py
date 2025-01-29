# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 17:39:54 2025

@author: HP
"""


import numpy as np
import astropy.units as u
from radiation import *
import matplotlib.pyplot as plt
from statistics import mean, pvariance, pstdev


# This is the calculation and plot of Fig. 5. We will change the masses and 
# see how the fluxes vary over the frequency. Here, it is important to mark
# that there are the three fluxes (S, C, B). They are a combination of formulas 
# 25, 30 and 38 taking into account 56 and the way the information is codified.

# The first mass is m = 5e7. Here we have the flux.

m_1 = 5e7
First_flux = Final_flux(m_1)

# The second mass is m = 5e8. Here we have the flux.

m_2 = 5e8
Second_flux = Final_flux(m_2)

# The third mass is m = 1e9. Here we have the flux.

m_3 = 1e9
Third_flux = Final_flux(m_3)

# The fourth mass is m = 3e9. Here we have the flux.

m_4 = 3e9
Fourth_flux = Final_flux(m_4)

plt.figure(figsize=(8,6))
plt.plot(np.log10(First_flux[0]), First_flux[2][0], color = 'b', label = "$m=5 x 10^{7}$") 
plt.plot(np.log10(First_flux[1]), First_flux[2][1], color = 'b')
plt.plot(np.log10(Second_flux[0]), Second_flux[2][0], color = 'r', label = "$m=5 x 10^{8}$") 
plt.plot(np.log10(Second_flux[1]), Second_flux[2][1], color = 'r')
plt.plot(np.log10(Third_flux[0]), Third_flux[2][0], color = 'g', label = "$m=1 x 10^{9}$") 
plt.plot(np.log10(Third_flux[1]), Third_flux[2][1], color = 'g')
plt.plot(np.log10(Fourth_flux[0]), Fourth_flux[2][0], color = 'm', label = "$m=3 x 10^{9}$") 
plt.plot(np.log10(Fourth_flux[1]), Fourth_flux[2][1], color = 'm')
plt.xlabel(r"$log(\nu \hspace{0.7} Hz)$")
plt.ylabel(r"$\log(\nu L_{\nu} \hspace{1} ergs \hspace{0.2} s^{-1})$")
plt.legend()
plt.show()



data_Flux_m_5e7 = np.loadtxt("data/Final_flux_m_5e7.txt", delimiter=",")
ref_v = data_Flux_m_5e7[:, 0]
ref_vLv = data_Flux_m_5e7[:, 1]


plt.plot(np.log10(First_flux[0]), First_flux[2][0], color = 'b', label = "$m=5 x 10^{7}$") 
plt.plot(np.log10(First_flux[1]), First_flux[2][1], color = 'b')
plt.plot(ref_v, ref_vLv, color = 'r', label = "Reference")
plt.xlabel(r"$log(\nu \hspace{0.7} Hz)$")
plt.ylabel(r"$\log(\nu L_{\nu} \hspace{1} ergs \hspace{0.2} s^{-1})$")
plt.legend()
plt.show()




