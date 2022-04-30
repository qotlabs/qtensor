import numpy as np
import matplotlib.pyplot as plt
from qtensor import Loader

loader = Loader('Results.xlsx')
sheet_name = 'Purity'

purity_D_0 = loader.read_data(sheet_name, 'A', 1, 10)
purity_D_2 = loader.read_data(sheet_name, 'B', 1, 10)
purity_D_4 = loader.read_data(sheet_name, 'C', 1, 10)
purity_D_6 = loader.read_data(sheet_name, 'D', 1, 10)
purity_D_8 = loader.read_data(sheet_name, 'E', 1, 10)
purity_D_10 = loader.read_data(sheet_name, 'F', 1, 10)

list_N = np.linspace(1, 10, 10)

fig, ax = plt.subplots(figsize=(11, 5.7))
# plt.plot(list_N, purity_D_0, lw=3, alpha=0.7, label=r'$N = 10, D = 0$')
# plt.plot(list_N, purity_D_2, lw=3, alpha=0.7, label=r'$N = 10, D = 2$')
# plt.plot(list_N, purity_D_4, lw=3, alpha=0.7, label=r'$N = 10, D = 4$')
# plt.plot(list_N, purity_D_6, lw=3, alpha=0.7, label=r'$N = 10, D = 6$')
# plt.plot(list_N, purity_D_8, lw=3, alpha=0.7, label=r'$N = 10, D = 8$')
# plt.plot(list_N, purity_D_10, lw=3, alpha=0.7, label=r'$N = 10, D = 10$')
# ax.bar(list_N, purity_D_0, width=0.1, lw=3, alpha=0.7, label=r'$N = 10, D = 0$')
ax.bar(list_N, purity_D_2, width=1, lw=3, alpha=0.2, color='green', label=r'$N = 10, D = 2$')
ax.bar(list_N, purity_D_4, width=1, lw=3, alpha=0.4, color='green', label=r'$N = 10, D = 4$')
ax.bar(list_N, purity_D_6, width=1, lw=3, alpha=0.6, color='green', label=r'$N = 10, D = 6$')
ax.bar(list_N, purity_D_8, width=1, lw=3, alpha=0.8, color='green', label=r'$N = 10, D = 8$')
ax.bar(list_N, purity_D_10, width=1, lw=3, alpha=1.0, color='green', label=r'$N = 10, D = 10$')

plt.tick_params(which='major', direction='in', labelsize=22)
plt.tick_params(which='minor', direction='in', labelsize=22)
plt.legend(loc='upper right', fontsize=22)
ax.minorticks_off()
# plt.xscale('log')
plt.xlim(1, 10)
plt.ylim(0.5, 1.02)
plt.xlabel(r'$n$', fontsize=22)
plt.ylabel(r'$\operatorname{Tr}\rho^2$', fontsize=22)
plt.show()
