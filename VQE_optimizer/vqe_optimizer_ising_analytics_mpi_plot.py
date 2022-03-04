import numpy as np
import matplotlib.pyplot as plt
from qtensor import Loader

loader = Loader('Results.xlsx')
sheet_name = 'VQE_Ising_Analytical'

number_of_iterations = 50

data_x = np.array(range(0, number_of_iterations, 1))
data_N = np.array(range(1, 51, 1))
data_theory_energy = -data_N / 2
data_energy = loader.read_data(sheet_name, 'A', 1, 2500)
data_energy_matrix = data_energy.reshape((len(data_N), number_of_iterations))
print(data_energy_matrix.shape)

data_energy_min = []
for n in data_N:
    data_energy_min.append(data_energy[n * number_of_iterations - 1])

data_energy_min = np.array(data_energy_min)

print(len(np.linspace(1 - 0.1, 1 + 0.1, number_of_iterations)))

fig, ax = plt.subplots()
# plt.plot(data_N, data_energy_min, lw=2, color='blue', alpha=0.5, label='Optimization')
# plt.plot(data_N, data_theory_energy, '--', lw=3, color='blue', alpha=0.5, label='Theory value')
for i, n in enumerate(data_N):
    plt.scatter(np.linspace(n - 0.1, n + 0.1, number_of_iterations), data_energy_matrix[i], color='blue', alpha=0.5)
plt.tick_params(which='major', direction='in', labelsize=16)
plt.tick_params(which='minor', direction='in', labelsize=16)

plt.legend(loc='upper right')
ax.minorticks_off()
# plt.yscale('log')
# plt.xlim(0, 19)
# plt.ylim(0.01, 1.1)
plt.xlabel('N', fontsize=15)
plt.ylabel(r'$E$', fontsize=15)
plt.show()
