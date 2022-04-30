import numpy as np
import matplotlib.pyplot as plt
from qtensor import Loader

N = 20

load = Loader('../Results.xlsx')
sheet_name = 'Multi_qubit_fidelity'
fid_result_two_10 = load.read_data(sheet_name, 'A', 1, 50)
fid_result_multi_10 = load.read_data(sheet_name, 'B', 1, 50)
fid_result_two_20 = load.read_data(sheet_name, 'C', 1, 50)
fid_result_multi_20 = load.read_data(sheet_name, 'D', 1, 50)
fid_result_two_50 = load.read_data(sheet_name, 'E', 1, 25)
fid_result_multi_50 = load.read_data(sheet_name, 'F', 1, 25)
data_x = np.array(range(0, 50, 1))

fid_mean_result_two_10 = np.array([fid_result_two_10[0]] + [np.exp(np.log(fid_result_two_10[0:d]).mean()) for d in
                                                            data_x[1:]])
fid_mean_result_two_20 = np.array([fid_result_two_20[0]] + [np.exp(np.log(fid_result_two_20[0:d]).mean()) for d in
                                                            data_x[1:]])
fid_mean_result_two_50 = np.array([fid_result_two_50[0]] + [np.exp(np.log(fid_result_two_50[0:d]).mean()) for d in
                                                            data_x[1:25]])

fid_multi_two_10 = np.array([fid_mean_result_two_10[d] ** ((N - 1) * d / 2) for d in data_x])
# fid_multi_two_20 = np.array([np.exp(np.log(fid_result_two_20[0:d] ** ((N - 1) / 2)).sum()) for d in data_x])
fid_multi_two_20 = np.array([fid_mean_result_two_20[d] ** ((N - 1) * d / 2) for d in data_x])
fid_multi_two_50 = np.array([fid_mean_result_two_50[d] ** ((N - 1) * d / 2) for d in data_x[0:25]])

fig, ax = plt.subplots()
# plt.plot(data_x, fid_result_two_10, lw=1, alpha=1, label=r'$N = 20, \chi = 10$')
plt.plot(data_x, fid_mean_result_two_10, '--', lw=3, alpha=1, color='red', label=r'$N = 20, \chi = 10$')
# plt.plot(data_x, fid_result_two_20, lw=1, alpha=1, label=r'$N = 20, \chi = 20$')
plt.plot(data_x, fid_mean_result_two_20, '--', lw=3, alpha=1, color='blue', label=r'$N = 20, \chi = 20$')
# plt.plot(data_x[0:25], fid_result_two_50, lw=1, alpha=1, label=r'$N = 20, \chi = 50$')
plt.plot(data_x[0:25], fid_mean_result_two_50, '--', lw=3, alpha=1, color='green', label=r'$N = 20, \chi = 50$')

plt.legend(loc='lower left')
ax.minorticks_off()
plt.xlim(0, 25)
plt.xlabel(r'$D$', fontsize=15)
plt.ylabel(r'$\langle f_n \rangle$', fontsize=15)
plt.show()

fig, ax = plt.subplots(figsize=(9, 5.7))
plt.plot(data_x, fid_result_multi_10, '--', lw=5, alpha=1, color='gray', label=r'$N = 20, \chi = 10$')
# plt.scatter(data_x, fid_multi_two_10, color='red')
plt.plot(data_x, fid_result_multi_20, '--', lw=5, alpha=1, color='blue', label=r'$N = 20, \chi = 20$')
# plt.scatter(data_x, fid_multi_two_20, color='blue')
plt.plot(data_x[0:25], fid_result_multi_50, '--', lw=5, alpha=1, color='red', label=r'$N = 20, \chi = 50$')
# plt.scatter(data_x[0:25], fid_multi_two_50, color='green')
plt.tick_params(which='major', direction='in', labelsize=22)
plt.tick_params(which='minor', direction='in', labelsize=22)
plt.legend(loc='lower left', fontsize=22)
ax.minorticks_off()
plt.xlim(0, 20)
plt.ylim(10 ** (-3), 1.0)
plt.yscale('log')
plt.xlabel(r'$D$', fontsize=22)
plt.ylabel(r'$F$', fontsize=22)
plt.show()
