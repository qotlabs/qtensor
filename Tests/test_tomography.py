import numpy as np
from qtensor import Info, DataModel

N = 5

info = Info()
data_model = DataModel(info)

data_model.gen_pure_state(N, 5)

print(data_model.state.r)
print(data_model.state.get_trace())
print(np.linalg.eigvals(np.array(data_model.state.get_full_matrix())))
print(np.sum(np.abs(np.array(data_model.state.get_full_matrix()) -
                    np.array(data_model.state.get_full_matrix()).T.conjugate()) ** 2))

data_model.gen_data(10, 1, 5, 5)

print(len(data_model.data_train))
print(len(data_model.data_test))
print(data_model.data_train[0][0].r)
print(data_model.data_test[0][0].r)

print(data_model.data_train[0][0].get_trace())
print(np.linalg.eigvals(np.array(data_model.data_train[0][0].get_full_matrix())))
print(np.sum(np.abs(np.array(data_model.data_train[0][0].get_full_matrix()) -
                    np.array(data_model.data_train[0][0].get_full_matrix()).T.conjugate()) ** 2))

print(data_model.get_mini_batch(5))
