import numpy as np
import matplotlib.pyplot as plt
from qtensor import Loader

loader = Loader('Results.xlsx')
sheet_name = 'CX'

nums_of_sample = 100
list_ranks = np.arange(1, 33, 1)

results_mean_exact = loader.read_data(sheet_name, 'A', 1, nums_of_sample * len(list_ranks)).reshape((nums_of_sample,
                                                                                                     len(list_ranks)))
results_mean = loader.read_data(sheet_name, 'B', 1, nums_of_sample * len(list_ranks)).reshape((nums_of_sample,
                                                                                               len(list_ranks)))
results_mean_ort = loader.read_data(sheet_name, 'C', 1, nums_of_sample * len(list_ranks)).reshape((nums_of_sample,
                                                                                                   len(list_ranks)))

res_error = np.mean(np.abs(results_mean - results_mean_exact), axis=0)
res_error_ort = np.mean(np.abs(results_mean_ort - results_mean_exact), axis=0)

print(res_error)
print(res_error_ort)

fig, ax = plt.subplots(figsize=(10, 7))
ax.bar(list_ranks - 0.125, res_error, width=0.25, lw=3, alpha=0.5, color='red', label='Without Ort')
ax.bar(list_ranks + 0.125, res_error_ort, width=0.25, lw=3, alpha=0.5, color='green', label='Ort')

plt.tick_params(which='major', direction='in', labelsize=22)
plt.tick_params(which='minor', direction='in', labelsize=22)
plt.legend(loc='upper right', fontsize=22)
ax.minorticks_off()
# plt.xscale('log')
plt.xlim(0.001, 16)
# plt.ylim(0.5, 1.02)
plt.xlabel(r'$\chi$', fontsize=22)
plt.ylabel(r'$|E - E_{\operatorname{exact}}|$', fontsize=22)
plt.show()


fig, ax = plt.subplots(figsize=(10, 7))
plt.scatter(list_ranks, results_mean[0], lw=5, alpha=1, color='red', label=r'$N = 50, D = 5$')
plt.plot(list_ranks, results_mean[0], lw=5, alpha=0.3, color='red')
plt.scatter(list_ranks, results_mean_ort[0], lw=5, alpha=1, color='green', label=r'$N = 50, D = 5$' + ' ort')
plt.plot(list_ranks, results_mean_ort[0], lw=5, alpha=0.3, color='green')
plt.plot(list_ranks, results_mean_exact[0], '--', lw=5, alpha=0.3, color='green', label='Theory value')

plt.tick_params(which='major', direction='in', labelsize=22)
plt.tick_params(which='minor', direction='in', labelsize=22)
plt.legend(loc='lower right', fontsize=22)
ax.minorticks_off()
# plt.xscale('log')
plt.xlim(0.001, 16)
# plt.ylim(0, 1.5)
plt.xlabel(r'$\chi$', fontsize=22)
plt.ylabel(r'$E$', fontsize=22)
plt.show()
