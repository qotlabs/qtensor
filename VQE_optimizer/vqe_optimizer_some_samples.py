import numpy as np
import matplotlib.pyplot as plt
from qtensor import Loader

loader = Loader('Results.xlsx')
sheet_name = 'VQE_Ising_Analytical'

number_of_iterations = 50

data_x = np.array(range(0, number_of_iterations, 1))
data_N = np.array(range(1, 21, 1))
data_theory_energy = -data_N / 2
data_energy_D1 = loader.read_data(sheet_name, 'A', 1, 1000)
data_energy_matrix_D1 = data_energy_D1.reshape((len(data_N), number_of_iterations))
data_energy_D2 = loader.read_data(sheet_name, 'B', 1, 1000)
data_energy_matrix_D2 = data_energy_D2.reshape((len(data_N), number_of_iterations))
data_energy_D3 = loader.read_data(sheet_name, 'C', 1, 1000)
data_energy_matrix_D3 = data_energy_D3.reshape((len(data_N), number_of_iterations))
data_energy_D3_chi_2 = loader.read_data(sheet_name, 'D', 1, 1000)
data_energy_matrix_D3_chi_2 = data_energy_D3_chi_2.reshape((len(data_N), number_of_iterations))

print(data_theory_energy[9])
fig, ax = plt.subplots(figsize=(9, 7))
# plt.plot(data_N, data_energy_min, lw=2, color='blue', alpha=0.5, label='Optimization')
# plt.plot(data_N, data_theory_energy, '--', lw=3, color='red', alpha=0.5, label='Theory value')
# plt.plot(np.linspace(1, number_of_iterations, number_of_iterations),
#          np.array([data_theory_energy[9]] * number_of_iterations), '--', lw=3, color='gray', alpha=0.5,
#          label='Theory value')
plt.plot(np.linspace(1, number_of_iterations, number_of_iterations), data_energy_matrix_D1[19] - data_theory_energy[19],
         '--', lw=3, color='blue',
         alpha=1, label=r'$D = 1$')
plt.plot(np.linspace(1, number_of_iterations, number_of_iterations), data_energy_matrix_D2[19] - data_theory_energy[19],
         '--', lw=3, color='green',
         alpha=1, label=r'$D = 2$')
plt.plot(np.linspace(1, number_of_iterations, number_of_iterations), data_energy_matrix_D3[19] - data_theory_energy[19],
         '--', lw=3, color='orange',
         alpha=1, label=r'$D = 3$')
plt.plot(np.linspace(1, number_of_iterations, number_of_iterations),
         data_energy_matrix_D3_chi_2[19] - data_theory_energy[19], '--', lw=3, color='red',
         alpha=1, label=r'$D = 3, \chi = 2$')
plt.tick_params(which='major', direction='in', labelsize=22)
plt.tick_params(which='minor', direction='in', labelsize=22)

plt.legend(loc='lower left', fontsize=22)
ax.minorticks_off()
plt.yscale('log')
# plt.xlim(0, 19)
# plt.ylim(0.01, 1.1)
plt.xlabel('Iterations', fontsize=22)
plt.ylabel(r'$E - E_0$', fontsize=22)
plt.show()
