import numpy as np
from qtensor import Info, DataModel, LearnModel

N = 5

info = Info()

data_model = DataModel(info)
learn_model = LearnModel(info)

data_model.gen_pure_state(N, 2)
learn_model.gen_start_state(N, 2)

print(data_model.state.r)
print(learn_model.omega.r)
print(learn_model.model.r)

data_model.gen_data(100, 1, 10, 1)

learn_model.optimize(data_model, 20, 1000, (2 ** N) ** 2)

print(learn_model.func_loss(learn_model.get_params(), data_model.data_test, (2 ** N) ** 2))

print(learn_model.model.get_trace())

print(data_model.data_test[0][1], learn_model.get_prob(data_model.data_test[0][0]))
