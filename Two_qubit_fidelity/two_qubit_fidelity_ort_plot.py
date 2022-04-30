import numpy as np
import matplotlib.pyplot as plt
from qtensor import Loader

load = Loader('../Results.xlsx')
sheet_name = 'Two_qubit_fidelity_ort'
fid_result_40_ort = load.read_data(sheet_name, 'A', 1, 150)
fid_result_40 = load.read_data(sheet_name, 'B', 1, 150)
fid_result_60_ort = load.read_data(sheet_name, 'C', 1, 150)
fid_result_60 = load.read_data(sheet_name, 'D', 1, 150)
data_x = np.array(range(0, 150, 1))


fid_mean_result_40_ort = np.array([fid_result_40_ort[0]] + [np.exp(np.log(fid_result_40_ort[0:d]).mean()) for d in
                                                            data_x[1:]])
fid_mean_result_40 = np.array([fid_result_40[0]] + [np.exp(np.log(fid_result_40[0:d]).mean()) for d in data_x[1:]])
fid_mean_result_60_ort = np.array([fid_result_60_ort[0]] + [np.exp(np.log(fid_result_60_ort[0:d]).mean()) for d in
                                                            data_x[1:]])
fid_mean_result_60 = np.array([fid_result_60[0]] + [np.exp(np.log(fid_result_60[0:d]).mean()) for d in data_x[1:]])

fig, ax = plt.subplots(figsize=(11, 5.7))
# plt.plot(data_x, fid_result_40, lw=1, alpha=1, color='red', label=r'$N = 40, \chi = 64$')
plt.plot(data_x, fid_mean_result_40_ort, '--', lw=5, alpha=1, color='red',
         label=r'$\langle f_n \rangle, N = 40, \chi = 64$' + ' ort')
plt.plot(data_x, fid_mean_result_40, '--', lw=5, alpha=0.3, color='red',
         label=r'$\langle f_n \rangle, N = 40, \chi = 64$')
# plt.plot(data_x, fid_result_60, lw=1, alpha=1, color='magenta', label=r'$N = 60, \chi = 64$')
plt.plot(data_x, fid_mean_result_60_ort, '--', lw=5, alpha=1, color='magenta',
         label=r'$\langle f_n \rangle, N = 60, \chi = 64$' + ' ort')
plt.plot(data_x, fid_mean_result_60, '--', lw=5, alpha=0.3, color='magenta',
         label=r'$\langle f_n \rangle, N = 60, \chi = 64$')

plt.tick_params(which='major', direction='in', labelsize=22)
plt.tick_params(which='minor', direction='in', labelsize=22)
plt.legend(loc='lower left', fontsize=22)
ax.minorticks_off()
plt.xlim(0, 150)
plt.xlabel(r'$D$', fontsize=22)
plt.ylabel(r'$\langle f_n \rangle$', fontsize=22)
plt.show()
