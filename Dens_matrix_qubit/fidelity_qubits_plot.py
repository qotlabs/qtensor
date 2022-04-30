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

fig, ax = plt.subplots(figsize=(11, 5.7))
ax.bar(list_N, fidelity_rank_1, width=0.5, lw=3, alpha=0.25, color='blue', label=r'$N = 10, \chi = 1$')
ax.bar(list_N, fidelity_rank_2, width=0.4, lw=3, alpha=0.2, color='blue', label=r'$N = 10, \chi = 2$')
ax.bar(list_N, fidelity_rank_4, width=0.3, lw=3, alpha=0.15, color='blue', label=r'$N = 10, \chi = 4$')
ax.bar(list_N, fidelity_rank_8, width=0.2, lw=3, alpha=0.1, color='blue', label=r'$N = 10, \chi = 8$')
ax.bar(list_N, fidelity_rank_16, width=0.1, lw=3, alpha=0.05, color='blue', label=r'$N = 10, \chi = 16$')
# ax.bar(list_N, fidelity_rank_32, lw=3, alpha=0.1, label=r'$N = 10, \chi = 32$')
plt.plot(list_N, fidelity_rank_1, lw=3, alpha=1.0, color='blue')
plt.plot(list_N, fidelity_rank_2, lw=3, alpha=0.8, color='blue')
plt.plot(list_N, fidelity_rank_4, lw=3, alpha=0.6, color='blue')
plt.plot(list_N, fidelity_rank_8, lw=3, alpha=0.4, color='blue')
plt.plot(list_N, fidelity_rank_16, lw=3, alpha=0.2, color='blue')

plt.tick_params(which='major', direction='in', labelsize=22)
plt.tick_params(which='minor', direction='in', labelsize=22)
plt.legend(loc='lower right', fontsize=22)
ax.minorticks_off()
# plt.xscale('log')
plt.xlim(1, 10)
plt.ylim(0.5, 1.02)
plt.xlabel(r'$n$', fontsize=22)
plt.ylabel(r'$F$', fontsize=22)
plt.show()
