import numpy as np
import matplotlib.pyplot as pd
import random


class SlotMachine:
    def __init__(self, true_rates):
        self.total_arm = true_rates.size
        self.total_choice_counter = 0
        self.total_win_counter = 0
        self.true_rates = true_rates
        self.choice_counter = np.zeros(self.total_arm, dtype=int)
        self.win_counter = np.zeros(self.total_arm, dtype=int)


    def pull_lever(self, machine_idx):
        self.choice_counter[machine_idx] += 1
        self.total_choice_counter += 1
        if self.true_rates[machine_idx] > random.random():
            self.win_counter[machine_idx] += 1
            self.total_win_counter += 1
            return True
        return False


    def calc_upper_confidence(self):
        return np.sqrt(2 * np.log(self.total_choice_counter) / self.choice_counter)
