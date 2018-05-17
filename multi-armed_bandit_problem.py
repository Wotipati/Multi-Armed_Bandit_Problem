import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import random
from slot_machine import SlotMachine



class Bandit:
    def __init__(self, machine_rate, iteration=10000, sample=10000, method=1):
        self.epsilon = 0.25
        self.total_iteration = iteration
        self.total_sample = sample
        self.total_arm = machine_rate.size
        self.true_rates = machine_rate
        self.choice_rate_history = np.zeros((iteration, self.total_arm))
        self.average_reward_history = np.zeros(iteration)
        self.method = method
        self.df_reward = pd.DataFrame()
        self.titles = ["Greedy strategy", "Epsiron-greedy strategy", "UBC1 strategy"]


    def trial(self):
        method = self.method
        print(self.titles[method])
        total_iteration = self.total_iteration
        for episode in range(self.total_sample):
            print("Episode {0} / {1}".format(episode+1, self.total_sample))
            slot_machine = SlotMachine(self.true_rates)

            for step in range(self.total_arm):
                choice = (step+episode)%self.total_arm
                for i in range(step, total_iteration):
                    self.choice_rate_history[i, choice] += 1

                if slot_machine.pull_lever(choice):
                    for i in range(step, total_iteration):
                        self.average_reward_history[i] += 1


            for step in range(self.total_arm, total_iteration):
                choice = self.choice_next_arm(slot_machine, method)
                for i in range(step, total_iteration):
                    self.choice_rate_history[i, choice] += 1

                if slot_machine.pull_lever(choice):
                    for i in range(step, total_iteration):
                        self.average_reward_history[i] += 1

        for i in range(total_iteration):
            self.choice_rate_history[i] /= i+1

        self.choice_rate_history /= self.total_sample
        self.average_reward_history /= np.arange(1, total_iteration+1, 1)
        self.make_df_reward_history()


    def choice_next_arm(self, machine, method):
        # Greedy strategy
        if method==0:
            obserbed_rates = machine.win_counter/machine.choice_counter
            cand = np.where(obserbed_rates == obserbed_rates.max())
            if cand[0].size==1:
                return cand[0][0]
            else:
                return np.random.choice(cand[0], 1)[0]


        # Epsilon-Greedy strategy
        elif method==1:
            # exploration
            if self.epsilon>random.random():
                return random.randint(0, self.total_arm-1)

            # exploition
            else:
                obserbed_rates = machine.win_counter/machine.choice_counter
                return np.argmax(obserbed_rates)


        # UBC1 strategy
        elif method==2:
            obserbed_rates = machine.win_counter/machine.choice_counter
            ucb_value = obserbed_rates + machine.calc_upper_confidence()
            return np.argmax(ucb_value)



    def visualize_choice_process(self):
        plt.style.use('ggplot')
        font = {'family' : 'meiryo'}
        matplotlib.rc('font', **font)
        df = pd.DataFrame(self.choice_rate_history, columns=self.true_rates)
        df.plot.area(alpha=0.4,figsize=(10,7))
        plt.title(self.titles[self.method], size=20)
        plt.xlabel("Iteration", size=18)
        plt.ylim([0.0, 1.0])
        plt.tick_params(labelsize=14)
        plt.legend(fontsize=16)


    def make_df_reward_history(self):
        self.df_reward = pd.DataFrame(self.average_reward_history, columns=[self.titles[self.method]])


    def visualize_reward_process(self):
        plt.style.use('ggplot')
        font = {'family' : 'meiryo'}
        matplotlib.rc('font', **font)
        self.df_reward.plot(alpha=0.4,figsize=(10,7))
        plt.title("Average reward", size=20)
        plt.xlabel("Iteration", size=18)
        plt.tick_params(labelsize=14)
        plt.legend(fontsize=16)


def main():
    # set parameter
    true_rate = np.array([0.5,0.4,0.3,0.2])  # winning rate
    iteration = 5000                        # Total round
    sample = 100                             # Number of trials

    bandit = Bandit(true_rate, iteration, sample, 0)
    bandit.trial()
    bandit.visualize_choice_process()

    bandit_epsilon = Bandit(true_rate, iteration, sample, 1)
    bandit_epsilon.trial()
    bandit_epsilon.visualize_choice_process()
    bandit.df_reward['Epsilon-greedy strategy'] = bandit_epsilon.average_reward_history

    bandit_ucb = Bandit(true_rate, iteration, sample, 2)
    bandit_ucb.trial()
    bandit_ucb.visualize_choice_process()
    bandit.df_reward['UCB1 strategy'] = bandit_ucb.average_reward_history

    bandit.visualize_reward_process()
    plt.show()


if __name__ == '__main__':
    main()
