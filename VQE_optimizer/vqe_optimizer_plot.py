import numpy as np
import matplotlib.pyplot as plt
from qtensor import Loader

loader = Loader('../Results.xlsx')
sheet_name = 'VQE'

data_x = np.array(range(0, 20, 1))
data_N_5_D_1 = loader.read_data(sheet_name, 'A', 1, 20)
data_N_5_D_2 = loader.read_data(sheet_name, 'B', 1, 20)
data_N_5_D_3 = loader.read_data(sheet_name, 'C', 1, 20)
data_N_5_D_4 = loader.read_data(sheet_name, 'D', 1, 20)
data_N_5_D_5 = loader.read_data(sheet_name, 'E', 1, 20)

data_min_energy = np.array([loader.read_data(sheet_name, 'A', 22, 22)] * 20)

fig, ax = plt.subplots()
plt.plot(data_x, (data_N_5_D_1 - data_min_energy[0]) / abs(data_min_energy[0]), lw=2, alpha=1, label=r'$N = 5, D = 1$')
plt.plot(data_x, (data_N_5_D_2 - data_min_energy[0]) / abs(data_min_energy[0]), lw=2, alpha=1, label=r'$N = 5, D = 2$')
plt.plot(data_x, (data_N_5_D_3 - data_min_energy[0]) / abs(data_min_energy[0]), lw=2, alpha=1, label=r'$N = 5, D = 3$')
plt.plot(data_x, (data_N_5_D_4 - data_min_energy[0]) / abs(data_min_energy[0]), lw=2, alpha=1, label=r'$N = 5, D = 4$')
plt.plot(data_x, (data_N_5_D_5 - data_min_energy[0]) / abs(data_min_energy[0]), lw=2, alpha=1, label=r'$N = 5, D = 5$')
# plt.plot(data_x, data_min_energy, '--', lw=3, alpha=1, label='Min energy')
plt.tick_params(which='major', direction='in', labelsize=16)
plt.tick_params(which='minor', direction='in', labelsize=16)

plt.legend(loc='upper right')
ax.minorticks_off()
plt.yscale('log')
plt.xlim(0, 19)
plt.ylim(0.01, 1.1)
plt.xlabel('Номер итерации', fontsize=15)
plt.ylabel(r'$\dfrac{E - E_{\min}}{|E_{\min}|}$', fontsize=15)
plt.show()

# loader = Loader('../Results.xlsx')
# sheet_name = 'VQE'
#
# data_x = np.array(range(0, 20, 1))
# data_N_10_D_1 = loader.read_data(sheet_name, 'F', 1, 20)
# data_N_15_D_1 = loader.read_data(sheet_name, 'G', 1, 20)
# data_N_20_D_1 = loader.read_data(sheet_name, 'H', 1, 20)
# data_N_30_D_1 = loader.read_data(sheet_name, 'I', 1, 20)
# data_N_50_D_1 = loader.read_data(sheet_name, 'J', 1, 20)
#
# data_min_energy = np.array([loader.read_data(sheet_name, 'F', 22, 22)] * 20)
#
# fig, ax = plt.subplots()
# plt.plot(data_x, data_N_10_D_1, lw=2, alpha=1, label=r'$N = 10, D = 1$')
# # plt.plot(data_x, data_N_15_D_1, lw=2, alpha=1, label=r'$N = 15, D = 1$')
# plt.plot(data_x, data_N_20_D_1, lw=2, alpha=1, label=r'$N = 20, D = 1$')
# plt.plot(data_x, data_N_30_D_1, lw=2, alpha=1, label=r'$N = 30, D = 1$')
# plt.plot(data_x, data_N_50_D_1, lw=2, alpha=1, label=r'$N = 50, D = 1$')
# # plt.plot(data_x, data_min_energy, '--', color='blue', lw=3, alpha=1, label='Min energy')
# plt.tick_params(which='major', direction='in', labelsize=16)
# plt.tick_params(which='minor', direction='in', labelsize=16)
#
# plt.legend(loc='upper right')
# ax.minorticks_off()
# # plt.yscale('log')
# plt.xlim(0, 19)
# # plt.ylim(0.01, 1.1)
# plt.xlabel('Номер итерации', fontsize=15)
# plt.ylabel(r'$E$', fontsize=15)
# plt.show()
