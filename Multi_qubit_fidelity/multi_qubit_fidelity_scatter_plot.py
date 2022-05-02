import numpy as np
import matplotlib.pyplot as plt
from qtensor import Loader

loader = Loader('Results.xlsx')
sheet_name = 'CX'

nums_of_sample = 10000
result_fid_CX = loader.read_data(sheet_name, 'A', 1, nums_of_sample)
result_fid_CX_ort = loader.read_data(sheet_name, 'B', 1, nums_of_sample)

sheet_name = 'CZ'

nums_of_sample = 10000
result_fid_CZ = loader.read_data(sheet_name, 'A', 1, nums_of_sample)
result_fid_CZ_ort = loader.read_data(sheet_name, 'B', 1, nums_of_sample)

fig, ax = plt.subplots(figsize=(8, 8))
# ax.bar(list_N, fidelity_rank_32, lw=3, alpha=0.1, label=r'$N = 10, \chi = 32$')
plt.scatter(result_fid_CZ[:100], result_fid_CZ_ort[:100], lw=3, alpha=0.5, color='blue', label=r'$C_Z$')
plt.scatter(result_fid_CX[:100], result_fid_CX_ort[:100], lw=3, alpha=0.5, color='red', label=r'$C_X$')
plt.plot(np.array([0, 1]), np.array([0, 1]), '--', lw=5, alpha=1.0, color='gray')

plt.tick_params(which='major', direction='in', labelsize=22)
plt.tick_params(which='minor', direction='in', labelsize=22)
plt.legend(loc='lower right', fontsize=22)
ax.minorticks_off()
# plt.xscale('log')
# plt.xlim(1, 10)
# plt.ylim(0.5, 1.02)
plt.xlabel(r'$F$', fontsize=22)
plt.ylabel(r'$F_{\operatorname{ort}}$', fontsize=22)
plt.show()


fig, ax = plt.subplots(figsize=(8, 8))
# ax.bar(list_N, fidelity_rank_32, lw=3, alpha=0.1, label=r'$N = 10, \chi = 32$')
plt.hist(result_fid_CZ, bins=50, lw=3, alpha=0.2, color='blue', label=r'$C_Z$')
plt.hist(result_fid_CZ_ort, bins=50, lw=3, alpha=0.5, color='blue', label=r'$C_Z$' + ' ort')
plt.hist(result_fid_CX, bins=50, lw=3, alpha=0.2, color='red', label=r'$C_X$')
plt.hist(result_fid_CX_ort, bins=50, lw=3, alpha=0.5, color='red', label=r'$C_X$' + ' ort')

plt.tick_params(which='major', direction='in', labelsize=22)
plt.tick_params(which='minor', direction='in', labelsize=22)
plt.legend(loc='lower right', fontsize=22)
ax.minorticks_off()
# plt.xscale('log')
# plt.xlim(1, 10)
# plt.ylim(0.5, 1.02)
plt.xlabel(r'$F$', fontsize=22)
plt.ylabel('Frequency', fontsize=22)
plt.show()
