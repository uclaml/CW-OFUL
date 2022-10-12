import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--corruption-level", default = 300, type = int)
args = parser.parse_args()

num_trials = 10

MO_result = []
W_result = []
O_result = []
RO_result = []
G_result = []
RB_result = []

for i in range(num_trials):
    with open("test_crr_{}_{}.txt".format(args.corruption_level, i), 'r') as json_file:
        data = json.load(json_file)
        MO_result.append(data['Multi-level-OFUL'])
        W_result.append(data['Weighted-OFUL'])
        O_result.append(data['OFUL'])
        RO_result.append(data['R-OFUL'])
        G_result.append(data['GREEDY'])
        RB_result.append(data['RobustBandit'])

MO_result = np.array(MO_result)
O_result = np.array(O_result)
RO_result = np.array(RO_result)
W_result = np.array(W_result)
G_result = np.array(G_result)
RB_result = np.array(RB_result)

MO_result_err = np.std(MO_result, axis = 0)
W_result_err = np.std(W_result, axis = 0)
O_result_err = np.std(O_result, axis = 0)
RO_result_err = np.std(RO_result, axis = 0)
G_result_err = np.std(G_result, axis = 0)
RB_result_err = np.std(RB_result, axis = 0)

MO_result = np.mean(MO_result, axis = 0)
W_result = np.mean(W_result, axis = 0)
O_result = np.mean(O_result, axis = 0)
RO_result = np.mean(RO_result, axis = 0)
G_result = np.mean(G_result, axis = 0)
RB_result = np.mean(RB_result, axis = 0)


print(MO_result)
print(W_result)
print(O_result)
print(RO_result)
print(G_result)
print(RB_result)


plt.plot(np.arange(O_result.shape[0]) * 100, O_result, color = 'r', label = 'OFUL', linestyle = '--')
plt.fill_between(np.arange(O_result.shape[0]) * 100, O_result - O_result_err, O_result + O_result_err, color = 'r', alpha = 0.1)


plt.plot(np.arange(RO_result.shape[0]) * 100, RO_result, color = 'b', label = 'Robust weighted OFUL', linestyle = '--')
plt.fill_between(np.arange(RO_result.shape[0]) * 100, RO_result - RO_result_err, RO_result + RO_result_err, color = 'b', alpha = 0.1)

plt.plot(np.arange(W_result.shape[0]) * 100, W_result, color = 'purple', label = 'Weighted OFUL', linestyle = '-.')
plt.fill_between(np.arange(W_result.shape[0]) * 100, W_result - W_result_err, W_result + W_result_err, color = 'purple', alpha = 0.1)
if args.corruption_level != 0:
    plt.plot(np.arange(G_result.shape[0]) * 100, G_result, color = 'orange', label = 'Greedy', linestyle = ':')
    plt.fill_between(np.arange(G_result.shape[0]) * 100, G_result - G_result_err, G_result + G_result_err, color = 'orange', alpha = 0.1)

plt.plot(np.arange(RB_result.shape[0]) * 100, RB_result, color = 'g', label = 'RobustBandit')
plt.fill_between(np.arange(RB_result.shape[0]) * 100, RB_result - RB_result_err, RB_result + RB_result_err, color = 'g', alpha = 0.2)

plt.plot(np.arange(MO_result.shape[0]) * 100, MO_result, color = 'k', label = 'Multi-level weighted OFUL')
plt.fill_between(np.arange(MO_result.shape[0]) * 100, MO_result - MO_result_err, MO_result + MO_result_err, color = 'k', alpha = 0.2)

plt.xlabel("Number of Rounds")
plt.ylabel("Total Regret")

plt.legend()

plt.savefig('corruption-level-' + str(args.corruption_level * 2) + '.pdf')
plt.savefig('corruption-level-' + str(args.corruption_level * 2) + '.png')

plt.show()