from scipy.optimize import linear_sum_assignment
from ThompsonSamplingOLA.Environment import Environment
from UCB import UCB_Matching
import matplotlib.pyplot as plt
import numpy as np


def run1():
    p = np.array([[1 / 4, 1, 1 / 4], [1 / 2, 1 / 4, 1 / 4], [1 / 4, 1 / 4, 1]])
    opt = linear_sum_assignment(-p)
    n_exp = 40
    T = 3000
    regret_ucb = np.zeros((n_exp, T))
    for e in range(n_exp):
        learner = UCB_Matching(p.size, *p.shape)
        rew_UCB = []
        opt_rew = []
        env = Environment(p.size, p)
        for t in range(T):
            pulled_arms = learner.pull_arm()
            rewards = env.round(pulled_arms)
            learner.update(pulled_arms, rewards)
            rew_UCB.append(rewards.sum())
            opt_rew.append(p[opt].sum())
        regret_ucb[e, :] = np.cumsum(opt_rew) - np.cumsum(rew_UCB)

    plt.figure(0)
    plt.plot(regret_ucb.mean(axis=0))
    plt.ylabel('Regret')
    plt.xlabel('t')
    plt.show()


def run2():
    p0 = np.array([[1 / 4, 1, 1 / 4], [1 / 2, 1 / 4, 1 / 4], [1 / 4, 1 / 4, 1]])
    p1 = np.array([[1, 1 / 4, 1 / 4], [1 / 2, 1 / 4, 1 / 4], [1 / 4, 1 / 4, 1]])
    p2 = np.array([[1, 1 / 4, 1 / 4], [1 / 2, 1, 1 / 4], [1 / 4, 1 / 4, 1]])
    p = [p0, p1, p2]
    T = 3000
    n_exp = 10
    regret_cusum = np.zeros((n_exp, T))
    regret_ucb = np.zeros((n_exp, T))
    detections = [[] for _ in reange(n_exp)]
    M = 100
    eps = 0.1
    h = np.log(T)*2
    for j in range(n_exp):
        e_UCB = Non_S