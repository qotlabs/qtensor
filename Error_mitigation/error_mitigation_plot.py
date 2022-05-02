import numpy as np
import matplotlib.pyplot as plt
from qtensor import Loader

N = 5
D = 10

loader = Loader('Results.xlsx')
sheet_name = 'Start'

nums_of_samples = 100

fid_start_start = loader.read_data(sheet_name, 'A', 1, nums_of_samples)
fid_finish_start = loader.read_data(sheet_name, 'B', 1, nums_of_samples)
fid_start_ort_start = loader.read_data(sheet_name, 'C', 1, nums_of_samples)
fid_finish_ort_start = loader.read_data(sheet_name, 'D', 1, nums_of_samples)

fig, ax = plt.subplots(figsize=(8, 8))
# plt.scatter(fid_start_start, fid_start_start, lw=1, alpha=0.5, color='red', label='Start')
# plt.scatter(fid_start_start, fid_finish_start, lw=1, alpha=0.5, color='green', label='Finish')
# plt.scatter(fid_start_ort_start, fid_start_ort_start, lw=1, alpha=0.5, color='red', label='Ort Start')
# plt.scatter(fid_start_ort_start, fid_finish_ort_start, lw=1, alpha=0.5, color='green', label='Ort Finish')
# plt.scatter(fid_start_start, fid_start_start, lw=1, alpha=0.5, color='red', label='Without Ort')
plt.scatter(fid_start_start, fid_finish_start, lw=1, alpha=0.5, color='red', label='Without Ort')
plt.scatter(fid_start_ort_start, fid_finish_ort_start, lw=1, alpha=0.5, color='blue', label='Ort')
plt.plot(np.array([0, 1]), np.array([0, 1]), '--', lw=5, alpha=1.0, color='gray')

plt.tick_params(which='major', direction='in', labelsize=22)
plt.tick_params(which='minor', direction='in', labelsize=22)
plt.legend(loc='lower right', fontsize=22)
ax.minorticks_off()
# plt.xscale('log')
plt.xlim(0.201, 0.799)
plt.ylim(0.201, 0.799)
plt.xlabel(r'$F$', fontsize=22)
plt.ylabel(r'$F_{\operatorname{mitigation}}$', fontsize=22)
plt.show()


loader = Loader('Results.xlsx')
sheet_name = 'Finish'

nums_of_samples = 100

fid_start_start = loader.read_data(sheet_name, 'A', 1, nums_of_samples)
fid_finish_start = loader.read_data(sheet_name, 'B', 1, nums_of_samples)
fid_start_ort_start = loader.read_data(sheet_name, 'C', 1, nums_of_samples)
fid_finish_ort_start = loader.read_data(sheet_name, 'D', 1, nums_of_samples)

fig, ax = plt.subplots(figsize=(8, 8))
# plt.scatter(fid_start_start, fid_start_start, lw=1, alpha=0.5, color='red', label='Start')
# plt.scatter(fid_start_start, fid_finish_start, lw=1, alpha=0.5, color='green', label='Finish')
# plt.scatter(fid_start_ort_start, fid_start_ort_start, lw=1, alpha=0.5, color='red', label='Ort Start')
# plt.scatter(fid_start_ort_start, fid_finish_ort_start, lw=1, alpha=0.5, color='green', label='Ort Finish')
# plt.scatter(fid_start_start, fid_start_start, lw=1, alpha=0.5, color='red', label='Without Ort')
# plt.scatter(fid_start_start, fid_start_ort_start, lw=1, alpha=0.5, color='green', label='Ort')
plt.scatter(fid_start_start, fid_finish_start, lw=1, alpha=0.5, color='red', label='Without Ort')
plt.scatter(fid_start_ort_start, fid_finish_ort_start, lw=1, alpha=0.5, color='blue', label='Ort')
plt.plot(np.array([0, 1]), np.array([0, 1]), '--', lw=5, alpha=1.0, color='gray')

plt.tick_params(which='major', direction='in', labelsize=22)
plt.tick_params(which='minor', direction='in', labelsize=22)
plt.legend(loc='lower right', fontsize=22)
ax.minorticks_off()
# plt.xscale('log')
plt.xlim(0.201, 0.799)
plt.ylim(0.201, 0.799)
plt.xlabel(r'$F$', fontsize=22)
plt.ylabel(r'$F_{\operatorname{mitigation}}$', fontsize=22)
plt.show()


loader = Loader('Results.xlsx')
sheet_name = 'Full'

nums_of_samples = 100

fid_start_start = loader.read_data(sheet_name, 'A', 1, nums_of_samples)
fid_finish_start = loader.read_data(sheet_name, 'B', 1, nums_of_samples)
fid_start_ort_start = loader.read_data(sheet_name, 'C', 1, nums_of_samples)
fid_finish_ort_start = loader.read_data(sheet_name, 'D', 1, nums_of_samples)

fig, ax = plt.subplots(figsize=(8, 8))
# plt.scatter(fid_start_start, fid_start_start, lw=1, alpha=0.5, color='red', label='Start')
# plt.scatter(fid_start_start, fid_finish_start, lw=1, alpha=0.5, color='green', label='Finish')
# plt.scatter(fid_start_ort_start, fid_start_ort_start, lw=1, alpha=0.5, color='red', label='Ort Start')
# plt.scatter(fid_start_ort_start, fid_finish_ort_start, lw=1, alpha=0.5, color='green', label='Ort Finish')
# plt.scatter(fid_start_start, fid_start_start, lw=1, alpha=0.5, color='red', label='Without Ort')
# plt.scatter(fid_start_start, fid_start_ort_start, lw=1, alpha=0.5, color='green', label='Ort')
plt.scatter(fid_start_start, fid_finish_start, lw=1, alpha=0.5, color='red', label='Without Ort')
plt.scatter(fid_start_ort_start, fid_finish_ort_start, lw=1, alpha=0.5, color='blue', label='Ort')
plt.plot(np.array([0, 1]), np.array([0, 1]), '--', lw=5, alpha=1.0, color='gray')

plt.tick_params(which='major', direction='in', labelsize=22)
plt.tick_params(which='minor', direction='in', labelsize=22)
plt.legend(loc='lower right', fontsize=22)
ax.minorticks_off()
# plt.xscale('log')
plt.xlim(0.201, 0.799)
plt.ylim(0.201, 0.799)
plt.xlabel(r'$F$', fontsize=22)
plt.ylabel(r'$F_{\operatorname{mitigation}}$', fontsize=22)
plt.show()


loader = Loader('Results.xlsx')
sheet_name = 'AllOneLayer'

nums_of_samples = 100

fid_start_start = loader.read_data(sheet_name, 'A', 1, nums_of_samples)
fid_finish_start = loader.read_data(sheet_name, 'B', 1, nums_of_samples)
fid_start_ort_start = loader.read_data(sheet_name, 'C', 1, nums_of_samples)
fid_finish_ort_start = loader.read_data(sheet_name, 'D', 1, nums_of_samples)

fig, ax = plt.subplots(figsize=(8, 8))
# plt.scatter(fid_start_start, fid_start_start, lw=1, alpha=0.5, color='red', label='Start')
# plt.scatter(fid_start_start, fid_finish_start, lw=1, alpha=0.5, color='green', label='Finish')
# plt.scatter(fid_start_ort_start, fid_start_ort_start, lw=1, alpha=0.5, color='red', label='Ort Start')
# plt.scatter(fid_start_ort_start, fid_finish_ort_start, lw=1, alpha=0.5, color='green', label='Ort Finish')
# plt.scatter(fid_start_start, fid_start_start, lw=1, alpha=0.5, color='red', label='Without Ort')
# plt.scatter(fid_start_start, fid_start_ort_start, lw=1, alpha=0.5, color='green', label='Ort')
plt.scatter(fid_start_start, fid_finish_start, lw=1, alpha=0.5, color='red', label='Without Ort')
plt.scatter(fid_start_ort_start, fid_finish_ort_start, lw=1, alpha=0.5, color='blue', label='Ort')
plt.plot(np.array([0, 1]), np.array([0, 1]), '--', lw=5, alpha=1.0, color='gray')

plt.tick_params(which='major', direction='in', labelsize=22)
plt.tick_params(which='minor', direction='in', labelsize=22)
plt.legend(loc='lower right', fontsize=22)
ax.minorticks_off()
# plt.xscale('log')
plt.xlim(0.201, 0.799)
plt.ylim(0.201, 0.799)
plt.xlabel(r'$F$', fontsize=22)
plt.ylabel(r'$F_{\operatorname{mitigation}}$', fontsize=22)
plt.show()

loader = Loader('Results.xlsx')
sheet_name = 'AllOneLayerMany'

nums_of_samples = 100

fid_start_start = loader.read_data(sheet_name, 'A', 1, nums_of_samples * D)
fid_finish_start = loader.read_data(sheet_name, 'B', 1, nums_of_samples * D)
fid_start_ort_start = loader.read_data(sheet_name, 'C', 1, nums_of_samples * D)
fid_finish_ort_start = loader.read_data(sheet_name, 'D', 1, nums_of_samples * D)

fig, ax = plt.subplots(figsize=(8, 8))
# plt.scatter(fid_start_start, fid_start_start, lw=1, alpha=0.5, color='red', label='Start')
# plt.scatter(fid_start_start, fid_finish_start, lw=1, alpha=0.5, color='green', label='Finish')
# plt.scatter(fid_start_ort_start, fid_start_ort_start, lw=1, alpha=0.5, color='red', label='Ort Start')
# plt.scatter(fid_start_ort_start, fid_finish_ort_start, lw=1, alpha=0.5, color='green', label='Ort Finish')
# plt.scatter(fid_start_start, fid_start_start, lw=1, alpha=0.5, color='red', label='Without Ort')
# plt.scatter(fid_start_start, fid_start_ort_start, lw=1, alpha=0.5, color='green', label='Ort')
plt.scatter(fid_start_start, fid_finish_start, lw=1, alpha=0.5, color='red', label='Without Ort')
plt.scatter(fid_start_ort_start, fid_finish_ort_start, lw=1, alpha=0.5, color='blue', label='Ort')
plt.plot(np.array([0, 1]), np.array([0, 1]), '--', lw=5, alpha=1.0, color='gray')

plt.tick_params(which='major', direction='in', labelsize=22)
plt.tick_params(which='minor', direction='in', labelsize=22)
plt.legend(loc='lower right', fontsize=22)
ax.minorticks_off()
# plt.xscale('log')
plt.xlim(0.501, 0.999)
plt.ylim(0.501, 0.999)
plt.xlabel(r'$F$', fontsize=22)
plt.ylabel(r'$F_{\operatorname{mitigation}}$', fontsize=22)
plt.show()


loader = Loader('Results.xlsx')
sheet_name = 'AllTwoLayer'

nums_of_samples = 100

fid_start_start = loader.read_data(sheet_name, 'A', 1, nums_of_samples)
fid_finish_start = loader.read_data(sheet_name, 'B', 1, nums_of_samples)
fid_start_ort_start = loader.read_data(sheet_name, 'C', 1, nums_of_samples)
fid_finish_ort_start = loader.read_data(sheet_name, 'D', 1, nums_of_samples)

fig, ax = plt.subplots(figsize=(8, 8))
# plt.scatter(fid_start_start, fid_start_start, lw=1, alpha=0.5, color='red', label='Start')
# plt.scatter(fid_start_start, fid_finish_start, lw=1, alpha=0.5, color='green', label='Finish')
# plt.scatter(fid_start_ort_start, fid_start_ort_start, lw=1, alpha=0.5, color='red', label='Ort Start')
# plt.scatter(fid_start_ort_start, fid_finish_ort_start, lw=1, alpha=0.5, color='green', label='Ort Finish')
# plt.scatter(fid_start_start, fid_start_start, lw=1, alpha=0.5, color='red', label='Without Ort')
# plt.scatter(fid_start_start, fid_start_ort_start, lw=1, alpha=0.5, color='green', label='Ort')
plt.scatter(fid_start_start, fid_finish_start, lw=1, alpha=0.5, color='red', label='Without Ort')
plt.scatter(fid_start_ort_start, fid_finish_ort_start, lw=1, alpha=0.5, color='blue', label='Ort')
plt.plot(np.array([0, 1]), np.array([0, 1]), '--', lw=5, alpha=1.0, color='gray')

plt.tick_params(which='major', direction='in', labelsize=22)
plt.tick_params(which='minor', direction='in', labelsize=22)
plt.legend(loc='lower right', fontsize=22)
ax.minorticks_off()
# plt.xscale('log')
plt.xlim(0.201, 0.799)
plt.ylim(0.201, 0.799)
plt.xlabel(r'$F$', fontsize=22)
plt.ylabel(r'$F_{\operatorname{mitigation}}$', fontsize=22)
plt.show()

loader = Loader('Results.xlsx')
sheet_name = 'AllTwoLayerMany'

nums_of_samples = 100

fid_start_start = loader.read_data(sheet_name, 'A', 1, nums_of_samples * (D // 2))
fid_finish_start = loader.read_data(sheet_name, 'B', 1, nums_of_samples * (D // 2))
fid_start_ort_start = loader.read_data(sheet_name, 'C', 1, nums_of_samples * (D // 2))
fid_finish_ort_start = loader.read_data(sheet_name, 'D', 1, nums_of_samples * (D // 2))

fig, ax = plt.subplots(figsize=(8, 8))
# plt.scatter(fid_start_start, fid_start_start, lw=1, alpha=0.5, color='red', label='Start')
# plt.scatter(fid_start_start, fid_finish_start, lw=1, alpha=0.5, color='green', label='Finish')
# plt.scatter(fid_start_ort_start, fid_start_ort_start, lw=1, alpha=0.5, color='red', label='Ort Start')
# plt.scatter(fid_start_ort_start, fid_finish_ort_start, lw=1, alpha=0.5, color='green', label='Ort Finish')
# plt.scatter(fid_start_start, fid_start_start, lw=1, alpha=0.5, color='red', label='Without Ort')
# plt.scatter(fid_start_start, fid_start_ort_start, lw=1, alpha=0.5, color='green', label='Ort')
plt.scatter(fid_start_start, fid_finish_start, lw=1, alpha=0.5, color='red', label='Without Ort')
plt.scatter(fid_start_ort_start, fid_finish_ort_start, lw=1, alpha=0.5, color='blue', label='Ort')
plt.plot(np.array([0, 1]), np.array([0, 1]), '--', lw=5, alpha=1.0, color='gray')

plt.tick_params(which='major', direction='in', labelsize=22)
plt.tick_params(which='minor', direction='in', labelsize=22)
plt.legend(loc='lower right', fontsize=22)
ax.minorticks_off()
# plt.xscale('log')
plt.xlim(0.501, 0.999)
plt.ylim(0.501, 0.999)
plt.xlabel(r'$F$', fontsize=22)
plt.ylabel(r'$F_{\operatorname{mitigation}}$', fontsize=22)
plt.show()
