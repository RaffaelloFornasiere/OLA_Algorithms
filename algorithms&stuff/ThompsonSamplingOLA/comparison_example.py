import matplotlib.pyplot as plt
from Environment import *
from Algorithms.Learners.TS_Learner import *
from Algorithms.Learners.Greedy_Learner import *

n_arms = 4
p = np.array([0.15, 0.1, 0.1, 0.35])
opt = p[3]
T = 300
n_experiments = 1000
ts_rewards_per_experiment = []
gr_rewards_per_experiment = []

for e in range(0, n_experiments):
    env = Environment(n_arms=n_arms, probabilities=p)
    ts_learner = TS_Learner(n_arms=n_arms)
    gr_learner = Greedy_Learner(n_arms=n_arms)
    for t in range(0, T):
        # Thompson Sampling Learners
        pulled_arm = ts_learner.pull_arm()
        reward = env.round(pulled_arm)
        ts_learner.update(pulled_arm, reward)

        # Greedy Learners
        pulled_arm = gr_learner.pull_arm()
        reward = env.round(pulled_arm)
        gr_learner.update(pulled_arm, reward)

    ts_rewards_per_experiment.append(ts_learner.collected_rewards)
    gr_rewards_per_experiment.append(gr_learner.collected_rewards)

print("end")
plt.figure(0)
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'r')
plt.plot(np.cumsum(np.mean(opt - gr_rewards_per_experiment, axis=0)), 'g')
plt.legend(["TS", "Greedy"])
plt.show()
