import numpy as np
import matplotlib.pyplot as plt
from qtensor import Load

load = Load('Results.xlsx')
sheet_name = 'Error_mitigation'

# rank_D_10 = load.read_data(sheet_name, 'A', 1, 11)
# mean_N_10 = load.read_data(sheet_name, 'B', 1, 12)
# mean_N_20 = load.read_data(sheet_name, 'D', 1, 12)
# mean_N_30 = load.read_data(sheet_name, 'F', 1, 12)
#
# data_x = np.array(rank_D_10)
#
# data_y_10 = np.array(mean_N_10[:-1])
# data_y_20 = np.array(mean_N_20[:-1])
# data_y_30 = np.array(mean_N_30[:-1])
#
# data_y_10_exact = np.array([mean_N_10[-1]] * len(data_x))
# data_y_20_exact = np.array([mean_N_20[-1]] * len(data_x))
# data_y_30_exact = np.array([mean_N_30[-1]] * len(data_x))
#
# fig, ax = plt.subplots()
# plt.scatter(data_x, data_y_10, lw=3, alpha=1, color='blue', label=r'$N = 10, D = 10$')
# plt.plot(data_x, data_y_10, lw=3, alpha=0.3, color='blue')
# plt.plot(data_x, data_y_10_exact, '--', lw=3, alpha=0.3, color='blue')
# plt.scatter(data_x, data_y_20, lw=3, alpha=1, color='red', label=r'$N = 20, D = 10$')
# plt.plot(data_x, data_y_20_exact, '--', lw=3, alpha=0.3, color='red')
# plt.plot(data_x, data_y_20, lw=3, alpha=0.3, color='red')
# plt.scatter(data_x, data_y_30, lw=3, alpha=1, color='green', label=r'$N = 30, D = 10$')
# plt.plot(data_x, data_y_30_exact, '--', lw=3, alpha=0.3, color='green')
# plt.plot(data_x, data_y_30, lw=3, alpha=0.3, color='green')
#
# plt.tick_params(which='major', direction='in', labelsize=16)
# plt.tick_params(which='minor', direction='in', labelsize=16)
# plt.legend(loc='lower right', fontsize=13)
# ax.minorticks_off()
# plt.xscale('log')
# plt.xlim(1, 1024)
# plt.ylim(-1, 1)
# plt.xlabel(r'$\chi$', fontsize=20)
# plt.ylabel(r'$\langle \psi |\hat{H}| \psi \rangle$', fontsize=20)
# plt.show()

# rank_D_5 = load.read_data(sheet_name, 'I', 1, 6)
# mean_N_30 = load.read_data(sheet_name, 'H', 1, 7)
# mean_N_40 = load.read_data(sheet_name, 'J', 1, 7)
# mean_N_50 = load.read_data(sheet_name, 'L', 1, 7)
# mean_N_100 = load.read_data(sheet_name, 'N', 1, 7)
# mean_N_200 = load.read_data(sheet_name, 'P', 1, 7)
#
# data_x = np.array(rank_D_5)
#
# data_y_30 = np.array(mean_N_30[:-1])
# data_y_40 = np.array(mean_N_40[:-1])
# data_y_50 = np.array(mean_N_50[:-1])
# data_y_100 = np.array(mean_N_100[:-1])
# data_y_200 = np.array(mean_N_200[:-1])
#
# data_y_30_exact = np.array([mean_N_30[-1]] * len(data_x))
# data_y_40_exact = np.array([mean_N_40[-1]] * len(data_x))
# data_y_50_exact = np.array([mean_N_50[-1]] * len(data_x))
# data_y_100_exact = np.array([mean_N_100[-1]] * len(data_x))
# data_y_200_exact = np.array([mean_N_200[-1]] * len(data_x))
#
# fig, ax = plt.subplots()
# # plt.scatter(data_x, data_y_30, lw=3, alpha=1, color='blue', label=r'$N = 30, D = 5$')
# # plt.plot(data_x, data_y_30, lw=3, alpha=0.3, color='blue')
# # plt.plot(data_x, data_y_30_exact, '--', lw=3, alpha=0.3, color='blue')
# # plt.scatter(data_x, data_y_40, lw=3, alpha=1, color='blue', label=r'$N = 40, D = 5$')
# # plt.plot(data_x, data_y_40_exact, '--', lw=3, alpha=0.3, color='blue')
# # plt.plot(data_x, data_y_40, lw=3, alpha=0.3, color='blue')
# plt.scatter(data_x, data_y_50, lw=3, alpha=1, color='blue', label=r'$N = 50, D = 5$')
# plt.plot(data_x, data_y_50_exact, '--', lw=3, alpha=0.3, color='blue')
# plt.plot(data_x, data_y_50, lw=3, alpha=0.3, color='blue')
# plt.scatter(data_x, data_y_100, lw=3, alpha=1, color='red', label=r'$N = 100, D = 5$')
# plt.plot(data_x, data_y_100_exact, '--', lw=3, alpha=0.3, color='red')
# plt.plot(data_x, data_y_100, lw=3, alpha=0.3, color='red')
# plt.scatter(data_x, data_y_200, lw=3, alpha=1, color='green', label=r'$N = 200, D = 5$')
# plt.plot(data_x, data_y_200_exact, '--', lw=3, alpha=0.3, color='green')
# plt.plot(data_x, data_y_200, lw=3, alpha=0.3, color='green')
#
# plt.tick_params(which='major', direction='in', labelsize=16)
# plt.tick_params(which='minor', direction='in', labelsize=16)
# plt.legend(loc='lower right', fontsize=13)
# ax.minorticks_off()
# plt.xscale('log')
# plt.xlim(1, 32)
# # plt.ylim(-1, 1)
# plt.xlabel(r'$\chi$', fontsize=20)
# plt.ylabel(r'$\langle \psi |\hat{H}| \psi \rangle$', fontsize=20)
# plt.show()

# rank_D_5 = load.read_data(sheet_name, 'Q', 1, 16)
# mean_N_200 = load.read_data(sheet_name, 'R', 1, 17)
#
# data_x = np.array(rank_D_5)
#
# data_y_200 = np.array(mean_N_200[:-1])
#
# data_y_200_exact = np.array([mean_N_200[-1]] * len(data_x))
#
# fig, ax = plt.subplots()
# # plt.scatter(data_x, data_y_30, lw=3, alpha=1, color='blue', label=r'$N = 30, D = 5$')
# # plt.plot(data_x, data_y_30, lw=3, alpha=0.3, color='blue')
# # plt.plot(data_x, data_y_30_exact, '--', lw=3, alpha=0.3, color='blue')
# # plt.scatter(data_x, data_y_40, lw=3, alpha=1, color='blue', label=r'$N = 40, D = 5$')
# # plt.plot(data_x, data_y_40_exact, '--', lw=3, alpha=0.3, color='blue')
# # plt.plot(data_x, data_y_40, lw=3, alpha=0.3, color='blue')
# plt.scatter(data_x, data_y_200, lw=3, alpha=1, color='blue', label=r'$N = 200, D = 5$')
# plt.plot(data_x, data_y_200_exact, '--', lw=3, alpha=0.3, color='blue')
# plt.plot(data_x, data_y_200, lw=3, alpha=0.3, color='blue')
#
#
# plt.tick_params(which='major', direction='in', labelsize=16)
# plt.tick_params(which='minor', direction='in', labelsize=16)
# plt.legend(loc='lower right', fontsize=13)
# ax.minorticks_off()
# plt.xscale('log')
# plt.xlim(1, 16)
# # plt.ylim(-1, 1)
# plt.xlabel(r'$\chi$', fontsize=20)
# plt.ylabel(r'$\langle \psi |\hat{H}| \psi \rangle$', fontsize=20)
# plt.show()

rank_D_5 = load.read_data(sheet_name, 'S', 1, 16)
mean_N_50 = load.read_data(sheet_name, 'T', 1, 34)

data_x = np.array(rank_D_5)

data_y_50 = np.array(mean_N_50[0:16])
data_y_50_ort = np.array(mean_N_50[17:33])

data_y_50_exact = np.array([mean_N_50[-1]] * len(data_x))

fig, ax = plt.subplots()
plt.scatter(data_x, data_y_50, lw=3, alpha=1, color='red', label=r'$N = 50, D = 5$')
plt.plot(data_x, data_y_50, lw=3, alpha=0.3, color='red')
plt.scatter(data_x, data_y_50_ort, lw=3, alpha=1, color='green', label=r'$N = 50, D = 5$' + ' ort')
plt.plot(data_x, data_y_50_ort, lw=3, alpha=0.3, color='green')
plt.plot(data_x, data_y_50_exact, '--', lw=3, alpha=0.3, color='green')

print(data_x)
print(data_y_50)
print(data_y_50_ort)
print(data_y_50_exact)

plt.tick_params(which='major', direction='in', labelsize=16)
plt.tick_params(which='minor', direction='in', labelsize=16)
plt.legend(loc='lower right', fontsize=13)
ax.minorticks_off()
plt.xscale('log')
plt.xlim(1, 16)
plt.ylim(0, 1.5)
plt.xlabel(r'$\chi$', fontsize=20)
plt.ylabel(r'$\langle \psi |\hat{H}| \psi \rangle$', fontsize=20)
plt.show()
