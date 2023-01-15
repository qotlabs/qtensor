import numpy as np
from qtensor import Info, DataModel, LearnModel

N = 5

info = Info()

print('Test DataModel')
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

print(data_model.state.r)
data_model.gen_mixed_state(N, 5)
print(data_model.state.r)

print('Test LearnModel')
learn_model = LearnModel(info)
learn_model.gen_start_state(N, 2)
print(learn_model.omega.r)
print(learn_model.model.r)

params = learn_model.get_params()
print(learn_model.omega.tt_cores[0])
print(params)
learn_model.set_params(params)
print(learn_model.omega.tt_cores[0])

data_model.gen_data(10, 5, 10, 5)
E = data_model.data_train[0][0]
p = data_model.data_train[0][1]
print(E, p)
print(data_model.get_prob(E))
# learn_model.model = data_model.state
print(learn_model.get_prob(E))
print(learn_model.omega.get_trace_product_matrix(E))
print(learn_model.model.get_trace_product_matrix(E))
learn_model.model = learn_model.omega.get_product_matrix(learn_model.omega.star())
print(learn_model.get_prob(E))

print()

mini_batch = data_model.get_mini_batch(2)
print(learn_model.func_loss(params, mini_batch, (2 ** N) ** 2))

print('Test gradient')
print(len(learn_model.grad_func_loss(params, mini_batch, (2 ** N) ** 2)))
print(learn_model.omega.r)
print(learn_model.omega.phys_ind_i)
print(learn_model.omega.phys_ind_j)
print(learn_model.grad_func_loss(params, mini_batch, (2 ** N) ** 2))
print(learn_model.grad_func_loss_test(params, mini_batch, (2 ** N) ** 2))
