import numpy as np
import matplotlib.pyplot as plt
from qtensor import Loader

N = 20

load = Loader('../Results.xlsx')
sheet_name = 'Multi_qubit_fidelity_ort'
fid_result_two_10_ort = load.read_data(sheet_name, 'A', 1, 40)
fid_result_multi_10_ort = load.read_data(sheet_name, 'B', 1, 40)
fid_result_two_10 = load.read_data(sheet_name, 'C', 1, 40)
fid_result_multi_10 = load.read_data(sheet_name, 'D', 1, 40)
fid_result_two_20_ort = load.read_data(sheet_name, 'E', 1, 40)
fid_result_multi_20_ort = load.read_data(sheet_name, 'F', 1, 40)
fid_result_two_20 = load.read_data(sheet_name, 'G', 1, 40)
fid_result_multi_20 = load.read_data(sheet_name, 'H', 1, 40)
fid_result_two_50_ort = load.read_data(sheet_name, 'I', 1, 40)
fid_result_multi_50_ort = load.read_data(sheet_name, 'J', 1, 40)
fid_result_two_50 = load.read_data(sheet_name, 'K', 1, 40)
fid_result_multi_50 = load.read_data(sheet_name, 'L', 1, 40)
data_x = np.array(range(0, 40, 1))

fid_mean_result_two_10_ort = np.array([fid_result_two_10[0]] + [np.exp(np.log(fid_result_two_10_ort[0:d]).mean()) for d
                                                                in data_x[1:]])
fid_mean_result_two_10 = np.array([fid_result_two_10[0]] + [np.exp(np.log(fid_result_two_10[0:d]).mean()) for d in
                                                            data_x[1:]])
fid_mean_result_two_20_ort = np.array([fid_result_two_20[0]] + [np.exp(np.log(fid_result_two_20_ort[0:d]).mean()) for d
                                                                in data_x[1:]])
fid_mean_result_two_20 = np.array([fid_result_two_20[0]] + [np.exp(np.log(fid_result_two_20[0:d]).mean()) for d in
                                                            data_x[1:]])
fid_mean_result_two_50_ort = np.array([fid_result_two_50[0]] + [np.exp(np.log(fid_result_two_50_ort[0:d]).mean()) for d
                                                                in data_x[1:]])
fid_mean_result_two_50 = np.array([fid_result_two_50[0]] + [np.exp(np.log(fid_result_two_50[0:d]).mean()) for d in
                                                            data_x[1:]])

fid_multi_two_10_ort = np.array([fid_mean_result_two_10_ort[d] ** ((N - 1) * d / 2) for d in data_x])
fid_multi_two_10 = np.array([fid_mean_result_two_10[d] ** ((N - 1) * d / 2) for d in data_x])
fid_multi_two_20_ort = np.array([fid_mean_result_two_20_ort[d] ** ((N - 1) * d / 2) for d in data_x])
fid_multi_two_20 = np.array([fid_mean_result_two_20[d] ** ((N - 1) * d / 2) for d in data_x])
fid_multi_two_50_ort = np.array([fid_mean_result_two_50_ort[d] ** ((N - 1) * d / 2) for d in data_x])
fid_multi_two_50 = np.array([fid_mean_result_two_50[d] ** ((N - 1) * d / 2) for d in data_x])

fig, ax = plt.subplots(figsize=(9, 5.7))
plt.plot(data_x, fid_result_multi_10_ort, '--', lw=5, alpha=1, color='gray', label=r'$N = 20, \chi = 10$' + ' ort')
# plt.scatter(data_x, fid_multi_two_10_ort, lw=0.5, alpha=0.5, color='red')
# plt.plot(data_x, fid_result_multi_10, '--', lw=5, alpha=1, color='gray', label=r'$N = 20, \chi = 10$')
# plt.scatter(data_x, fid_multi_two_10, alpha=0.5, color='red')
plt.plot(data_x, fid_result_multi_20_ort, '--', lw=5, alpha=1, color='blue', label=r'$N = 20, \chi = 20$' + ' ort')
# plt.scatter(data_x, fid_multi_two_20_ort, lw=0.5, alpha=0.5, color='blue')
# plt.plot(data_x, fid_result_multi_20, '--', lw=5, alpha=1, color='blue', label=r'$N = 20, \chi = 20$')
# plt.scatter(data_x, fid_multi_two_20, alpha=0.5, color='blue')
plt.plot(data_x, fid_result_multi_50_ort, '--', lw=5, alpha=1, color='red', label=r'$N = 20, \chi = 50$' + ' ort')
# plt.scatter(data_x, fid_multi_two_50_ort, lw=0.5, alpha=0.5, color='green')
# plt.plot(data_x, fid_result_multi_50, '--', lw=5, alpha=1, color='red', label=r'$N = 20, \chi = 50$')
# plt.scatter(data_x, fid_multi_two_50, alpha=0.5, color='green')

plt.legend(loc='lower left', fontsize=22)
ax.minorticks_off()
plt.xlim(0, 20)
plt.ylim(10 ** (-3), 1.0)
plt.yscale('log')
plt.tick_params(which='major', direction='in', labelsize=22)
plt.tick_params(which='minor', direction='in', labelsize=22)
plt.xlabel(r'$D$', fontsize=22)
plt.ylabel(r'$F$', fontsize=22)
plt.show()
