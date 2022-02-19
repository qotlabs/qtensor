from qtensor import IsingHamAnalytical, Info, Gates

N = 8
info = Info()
gates = Gates(info)
ham = IsingHamAnalytical(N, gates, info)
print(ham.get_min_energy_analytical())
print(ham.get_min_energy())
