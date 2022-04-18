import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List
import platform

# this code is from this tutorial
# https://www.youtube.com/watch?v=D8bofQKCUls&t=333s


class Restaurant:
    def __init__(self, mu, sigma):
        self.sum_satisfaction = 0
        self.n = 0
        self.mu = mu
        self.sigma = sigma

    def get_satisfaction_from_true_distribution(self):
        s = np.random.normal(self.mu, self.sigma)
        self.n += 1
        self.sum_satisfaction += s
        return s


class RestaurantThompsonSampler(Restaurant):
    def __init__(self, mu, sigma):
        self.prior_mu_of_mu = 0
        self.prior_sigma_of_mu = 10000

        self.post_mu_of_mu = self.prior_mu_of_mu
        self.post_sigma_of_mu = self.prior_sigma_of_mu

        super().__init__(mu, sigma)

    def get_mu_from_current_distribution(self):
        samp_mu = np.random.normal(self.post_mu_of_mu, self.post_sigma_of_mu)
        return samp_mu

    def update_current_distribution(self):
        self.post_sigma_of_mu = np.sqrt((1 / self.prior_sigma_of_mu ** 2 + self.n / self.sigma ** 2) ** -1)
        self.post_mu_of_mu = (self.post_sigma_of_mu ** 2) * ((self.prior_mu_of_mu / self.prior_sigma_of_mu ** 2) +
                                                             (self.sum_satisfaction / self.sigma ** 2))


def draw_distributions(restaurants: List[RestaurantThompsonSampler], i):
    for r in restaurants:
        samps = np.random.normal(r.post_mu_of_mu, r.post_sigma_of_mu, 10000)
        sns.kdeplot(samps, shade=True)
    plt.title("Iteration %s" % (i + 1), fontsize=16)
    plt.xlim(-10, 10)
    plt.xlabel('Average Satisfaction', fontsize=20)
    plt.ylabel('Density', fontsize=20)
    plt.draw()
    plt.pause(0.0001)
    plt.clf()


def run():
    num_restaurants = 10
    spacing = 0.33
    global_restaurants = [RestaurantThompsonSampler(i * spacing, 1) for i in range(1, num_restaurants + 1)]
    for i in range(2000):
        if i < 10 or (i < 100 and (i+1) % 10 == 0) or ((i+1) % 100 == 0):
            print(i)
            # draw_distributions(global_restaurants, i)
            if i > 0:
                plt.show()
            else:
                plt.plot()

        # get a sample for each posterior
        post_samps = [r.get_mu_from_current_distribution() for r in global_restaurants]

        # index of distribution with the highest satisfaction
        chosen_idx = post_samps.index(max(post_samps))

        # get a new sample from that distribution
        s = global_restaurants[chosen_idx].get_satisfaction_from_true_distribution()

        # update that distribution posterior
        global_restaurants[chosen_idx].update_current_distribution()

