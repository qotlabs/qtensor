import numpy as np
import matplotlib.pyplot as plt
from qtensor import Loader

loader = Loader('Results.xlsx')
sheet_name = 'Fidelity_qubits'

fidelity_rank_1 = loader.read_data(sheet_name, 'A', 1, 10)
fidelity_rank_2 = loader.read_data(sheet_name, 'B', 1, 10)
fidelity_rank_4 = loader.read_data(sheet_name, 'C', 1, 10)
fidelity_rank_8 = loader.read_data(sheet_name, 'D', 1, 10)
fidelity_rank_16 = loader.read_data(sheet_name, 'E', 1, 10)
fidelity_rank_32 = loader.read_data(sheet_name, 'F', 1, 10)

list_N = np.linspace(1, 10, 10)

fig, ax = plt.subplots()
plt.plot(list_N, fidelity_rank_1, lw=3, alpha=0.7, label=r'$N = 10, \chi = 1$')
plt.plot(list_N, fidelity_rank_2, lw=3, alpha=0.7, label=r'$N = 10, \chi = 2$')
plt.plot(list_N, fidelity_rank_4, lw=3, alpha=0.7, label=r'$N = 10, \chi = 4$')
plt.plot(list_N, fidelity_rank_8, lw=3, alpha=0.7, label=r'$N = 10, \chi = 8$')
plt.plot(list_N, fidelity_rank_16, lw=3, alpha=0.7, label=r'$N = 10, \chi = 16$')
plt.plot(list_N, fidelity_rank_32, lw=3, alpha=0.7, label=r'$N = 10, \chi = 32$')

plt.tick_params(which='major', direction='in', labelsize=16)
plt.tick_params(which='minor', direction='in', labelsize=16)
plt.legend(loc='lower right', fontsize=13)
ax.minorticks_off()
# plt.xscale('log')
plt.xlim(1, 10)
plt.ylim(0.5, 1.02)
plt.xlabel(r'$n$', fontsize=20)
plt.ylabel(r'$F$', fontsize=20)
plt.show()
