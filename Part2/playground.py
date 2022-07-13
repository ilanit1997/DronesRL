import numpy as np

# epsilon-greedy policy
e_sample = np.random.random()
curr_step = 2
start_epsilon, end_epsilon, n_steps = 1, 0.01, 30
eps_list = np.linspace(start_epsilon, end_epsilon, n_steps)
curr_eps = eps_list[curr_step]
random = np.random.binomial(1, curr_eps, size=1)
print(random)

random = 0 if 1 == 0 else random
print(random)

if random:
    print('explore')
else:

    print('exploit')
# if e_sample <= eps_list[curr_step]:
#     print('explore')
#     action = np.random.randint(0, 10)
# else:
#     print('exploit')