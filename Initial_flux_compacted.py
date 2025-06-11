import numpy as np
import astropy.units as u
from radiation import *
import matplotlib.pyplot as plt
from statistics import mean, pvariance, pstdev
from astropy.table import Table, Column, join
import pandas as pd

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
#Figure_initial_flux(m, m_dot_1, ref_m_dot_1, ref_T_e_1, ax=None)


# The second mass ratio is m_dot=6e-4, where there is the reference and the implementation

data_Flux_m_6e4 = np.loadtxt("data/Flux_m_6e-4.txt", delimiter=",")
ref_m_dot_2 = data_Flux_m_6e4[:, 0]
ref_T_e_2 = data_Flux_m_6e4[:, 1]
m_dot_2 = 6e-4
#Figure_initial_flux(m, m_dot_2, ref_m_dot_2, ref_T_e_2, ax=None)


# # The third mass ratio is m_dot=12e-4, where there is the reference and the implementation

data_Flux_m_12e4 = np.loadtxt("data/Flux_m_12e-4.txt", delimiter=",")
ref_m_dot_3 = data_Flux_m_12e4[:, 0]
ref_T_e_3 = data_Flux_m_12e4[:, 1]
m_dot_3 = 12e-4
#Figure_initial_flux(m, m_dot_3, ref_m_dot_3, ref_T_e_3, ax=None)

# # The fourth mass ratio is m_dot=24e-4, where there is the reference and the implementation

data_Flux_m_24e4 = np.loadtxt("data/Flux_m_24e-4.txt", delimiter=",")
ref_m_dot_4 = data_Flux_m_24e4[:, 0]
ref_T_e_4 = data_Flux_m_24e4[:, 1]
m_dot_4 = 24e-4
#Figure_initial_flux(m, m_dot_4, ref_m_dot_4, ref_T_e_4, ax=None)


# # Now,what we do is to present the four implementations in just on plot.


fig, ax = plt.subplots(figsize=(10, 8))
Figure_initial_flux(m, m_dot_1, ref_m_dot_1, ref_T_e_1, ax=ax, color = 'blue')
Figure_initial_flux(m, m_dot_2, ref_m_dot_2, ref_T_e_2, ax=ax, color = 'purple')
Figure_initial_flux(m, m_dot_3, ref_m_dot_3, ref_T_e_3, ax=ax, color = 'green')
Figure_initial_flux(m, m_dot_4, ref_m_dot_4, ref_T_e_4, ax=ax, color = 'black')
ax.legend()
plt.savefig("Initial_flux_four_cases.png", dpi=300, bbox_inches='tight')
plt.show()



cigale_file = f'NGC4261_table'
cigale = Table.read(f'{cigale_file}')
cigale=cigale.to_pandas()
df = pd.DataFrame(cigale)
t = Table.from_pandas(df)

Luminosity = 4*np.pi*df['col3']*np.power(35e6*(3.0856e18),2)

fig, ax = plt.subplots(figsize=(8, 6))
Figure_initial_flux_woref(m, m_dot_1, ax=ax, color = 'blue', label = 'm_dot=3e-4')
Figure_initial_flux_woref(m, m_dot_2, ax=ax, color = 'purple', label='m_dot=6e-4')
Figure_initial_flux_woref(m, m_dot_3, ax=ax, color = 'green', label='m_dot=12e-4')
Figure_initial_flux_woref(m, m_dot_4, ax=ax, color = 'red', label='m_dot=24e-4')
ax.plot(np.log10(df['col1']), np.log10(Luminosity), 'o', markersize=1)
ax.legend()
plt.show()





# plt.figure(figsize=(8,6))
# plt.plot(np.log10(df['col1']), np.log10(df['col3']), 'o', markersize=1)
# plt.xlim(8,21)
# plt.show()
