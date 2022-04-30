import numpy as np
import matplotlib.pyplot as plt
from qtensor import Loader

loader = Loader('Results.xlsx')
sheet_name = 'Start'

nums_of_samples = 100

fid_start_start = loader.read_data(sheet_name, 'A', 1, nums_of_samples)
fid_finish_start = loader.read_data(sheet_name, 'B', 1, nums_of_samples)
fid_start_ort_start = loader.read_data(sheet_name, 'C', 1, nums_of_samples)
fid_finish_ort_start = loader.read_data(sheet_name, 'D', 1, nums_of_samples)

fig, ax = plt.subplots()
# plt.scatter(fid_start_start, fid_start_start, lw=1, alpha=0.5, color='red', label='Start')
# plt.scatter(fid_start_start, fid_finish_start, lw=1, alpha=0.5, color='green', label='Finish')
# plt.scatter(fid_start_ort_start, fid_start_ort_start, lw=1, alpha=0.5, color='red', label='Ort Start')
# plt.scatter(fid_start_ort_start, fid_finish_ort_start, lw=1, alpha=0.5, color='green', label='Ort Finish')
plt.scatter(fid_start_start, fid_start_start, lw=1, alpha=0.5, color='red', label='Without Ort')
plt.scatter(fid_start_start, fid_start_ort_start, lw=1, alpha=0.5, color='green', label='Ort')

plt.tick_params(which='major', direction='in', labelsize=16)
plt.tick_params(which='minor', direction='in', labelsize=16)
plt.legend(loc='upper right', fontsize=13)
ax.minorticks_off()
# plt.xscale('log')
# plt.xlim(1, 10)
# plt.ylim(0.5, 1.02)
plt.xlabel(r'$F$', fontsize=20)
plt.ylabel(r'$F$', fontsize=20)
plt.show()


loader = Loader('Results.xlsx')
sheet_name = 'Finish'

nums_of_samples = 100

fid_start_start = loader.read_data(sheet_name, 'A', 1, nums_of_samples)
fid_finish_start = loader.read_data(sheet_name, 'B', 1, nums_of_samples)
fid_start_ort_start = loader.read_data(sheet_name, 'C', 1, nums_of_samples)
fid_finish_ort_start = loader.read_data(sheet_name, 'D', 1, nums_of_samples)

fig, ax = plt.subplots()
# plt.scatter(fid_start_start, fid_start_start, lw=1, alpha=0.5, color='red', label='Start')
# plt.scatter(fid_start_start, fid_finish_start, lw=1, alpha=0.5, color='green', label='Finish')
# plt.scatter(fid_start_ort_start, fid_start_ort_start, lw=1, alpha=0.5, color='red', label='Ort Start')
# plt.scatter(fid_start_ort_start, fid_finish_ort_start, lw=1, alpha=0.5, color='green', label='Ort Finish')
plt.scatter(fid_start_start, fid_start_start, lw=1, alpha=0.5, color='red', label='Without Ort')
plt.scatter(fid_start_start, fid_start_ort_start, lw=1, alpha=0.5, color='green', label='Ort')

plt.tick_params(which='major', direction='in', labelsize=16)
plt.tick_params(which='minor', direction='in', labelsize=16)
plt.legend(loc='upper right', fontsize=13)
ax.minorticks_off()
# plt.xscale('log')
# plt.xlim(1, 10)
# plt.ylim(0.5, 1.02)
plt.xlabel(r'$F$', fontsize=20)
plt.ylabel(r'$F$', fontsize=20)
plt.show()


loader = Loader('Results.xlsx')
sheet_name = 'AllOneLayer'

nums_of_samples = 100

fid_start_start = loader.read_data(sheet_name, 'A', 1, nums_of_samples)
fid_finish_start = loader.read_data(sheet_name, 'B', 1, nums_of_samples)
fid_start_ort_start = loader.read_data(sheet_name, 'C', 1, nums_of_samples)
fid_finish_ort_start = loader.read_data(sheet_name, 'D', 1, nums_of_samples)

fig, ax = plt.subplots()
plt.scatter(fid_start_start, fid_start_start, lw=1, alpha=0.5, color='red', label='Start')
plt.scatter(fid_start_start, fid_finish_start, lw=1, alpha=0.5, color='green', label='Finish')
# plt.scatter(fid_start_ort_start, fid_start_ort_start, lw=1, alpha=0.5, color='red', label='Ort Start')
# plt.scatter(fid_start_ort_start, fid_finish_ort_start, lw=1, alpha=0.5, color='green', label='Ort Finish')
# plt.scatter(fid_start_start, fid_start_start, lw=1, alpha=0.5, color='red', label='Without Ort')
# plt.scatter(fid_start_start, fid_start_ort_start, lw=1, alpha=0.5, color='green', label='Ort')

plt.tick_params(which='major', direction='in', labelsize=16)
plt.tick_params(which='minor', direction='in', labelsize=16)
plt.legend(loc='upper right', fontsize=13)
ax.minorticks_off()
# plt.xscale('log')
# plt.xlim(1, 10)
# plt.ylim(0.5, 1.02)
plt.xlabel(r'$F$', fontsize=20)
plt.ylabel(r'$F$', fontsize=20)
plt.show()


loader = Loader('Results.xlsx')
sheet_name = 'AllTwoLayer'

nums_of_samples = 100

fid_start_start = loader.read_data(sheet_name, 'A', 1, nums_of_samples)
fid_finish_start = loader.read_data(sheet_name, 'B', 1, nums_of_samples)
fid_start_ort_start = loader.read_data(sheet_name, 'C', 1, nums_of_samples)
fid_finish_ort_start = loader.read_data(sheet_name, 'D', 1, nums_of_samples)

fig, ax = plt.subplots()
# plt.scatter(fid_start_start, fid_start_start, lw=1, alpha=0.5, color='red', label='Start')
# plt.scatter(fid_start_start, fid_finish_start, lw=1, alpha=0.5, color='green', label='Finish')
# plt.scatter(fid_start_ort_start, fid_start_ort_start, lw=1, alpha=0.5, color='red', label='Ort Start')
# plt.scatter(fid_start_ort_start, fid_finish_ort_start, lw=1, alpha=0.5, color='green', label='Ort Finish')
plt.scatter(fid_start_start, fid_start_start, lw=1, alpha=0.5, color='red', label='Without Ort')
plt.scatter(fid_start_start, fid_start_ort_start, lw=1, alpha=0.5, color='green', label='Ort')

plt.tick_params(which='major', direction='in', labelsize=16)
plt.tick_params(which='minor', direction='in', labelsize=16)
plt.legend(loc='upper right', fontsize=13)
ax.minorticks_off()
# plt.xscale('log')
# plt.xlim(1, 10)
# plt.ylim(0.5, 1.02)
plt.xlabel(r'$F$', fontsize=20)
plt.ylabel(r'$F$', fontsize=20)
plt.show()


loader = Loader('Results.xlsx')
sheet_name = 'Full'

nums_of_samples = 100

fid_start_start = loader.read_data(sheet_name, 'A', 1, nums_of_samples)
fid_finish_start = loader.read_data(sheet_name, 'B', 1, nums_of_samples)
fid_start_ort_start = loader.read_data(sheet_name, 'C', 1, nums_of_samples)
fid_finish_ort_start = loader.read_data(sheet_name, 'D', 1, nums_of_samples)

fig, ax = plt.subplots()
# plt.scatter(fid_start_start, fid_start_start, lw=1, alpha=0.5, color='red', label='Start')
# plt.scatter(fid_start_start, fid_finish_start, lw=1, alpha=0.5, color='green', label='Finish')
plt.scatter(fid_start_ort_start, fid_start_ort_start, lw=1, alpha=0.5, color='red', label='Ort Start')
plt.scatter(fid_start_ort_start, fid_finish_ort_start, lw=1, alpha=0.5, color='green', label='Ort Finish')
# plt.scatter(fid_start_start, fid_start_start, lw=1, alpha=0.5, color='red', label='Without Ort')
# plt.scatter(fid_start_start, fid_start_ort_start, lw=1, alpha=0.5, color='green', label='Ort')

plt.tick_params(which='major', direction='in', labelsize=16)
plt.tick_params(which='minor', direction='in', labelsize=16)
plt.legend(loc='upper right', fontsize=13)
ax.minorticks_off()
# plt.xscale('log')
# plt.xlim(1, 10)
# plt.ylim(0.5, 1.02)
plt.xlabel(r'$F$', fontsize=20)
plt.ylabel(r'$F$', fontsize=20)
plt.show()
