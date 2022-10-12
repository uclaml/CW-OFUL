import numpy as np
import os
import argparse
import random
import math
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dim", default = 20, type = int)
    parser.add_argument("--range", default = .5, type = float)
    parser.add_argument("--corruption-level", default = 0, type = float)
    parser.add_argument("--num-rounds", default = 10000, type = int)
    parser.add_argument("--num-trials", default = 10, type = int)
    parser.add_argument("--seed", type=int, default = 201912)
    parser.add_argument("--norm-a", default = 1., type = float)
    parser.add_argument("--norm-b", default = 1., type = float)
    parser.add_argument("--num-actions", default = 20, type = int)
    args = parser.parse_args()
    return args

def init_vector(dim, norm):
    vec = 2 * np.random.rand(dim) - 1.
    return vec * norm / math.sqrt(dim)

def init_arms(dim, norm, num):
    decision = np.random.rand(num, dim)
    for i in range(num):
        decision[i] = init_vector(dim, norm)
    return decision

class M_OFUL(): 
    def __init__(self, args, lambda_init = 1., C = 0.05):
        self.ell_max = int(math.log(args.num_rounds * 2, 2) + 1)
        self.dim = args.hidden_dim
        self.sub_Sigma = np.zeros((self.ell_max + 1, self.dim, self.dim))
        for i in range(self.ell_max + 1):
            self.sub_Sigma[i] = np.identity(self.dim) * lambda_init
        self.gl_Sigma = np.identity(self.dim) * lambda_init
        self.gl_mu = np.zeros(self.dim)
        self.sub_mu = np.zeros((self.ell_max + 1, self.dim))
        self.gl_bbb = np.zeros(self.dim)
        self.sub_bbb = np.zeros((self.ell_max + 1, self.dim))
        self.noise_range = args.range
        self.crr = args.corruption_level
        self.num_rounds = args.num_rounds
        self.num_trials = args.num_trials
        self.num_actions = args.num_actions
        self.lambda_init = lambda_init
        self.args = args
        self.levels = [i for i in range(1, self.ell_max + 1)]
        self.levels_weight = [2**(-ell) for ell in range(1, self.ell_max + 1)]
        self.levels_weight[0] += 2**(-self.ell_max)
        self.C = C

    
    def calcuBeta(self, t_round, ell, delta = 0.033):
        log_term_1 = math.sqrt(math.log(1 + t_round * self.args.norm_a**2 / ((self.noise_range + 1)**2 * self.lambda_init)))
        log_term_2 = math.log(4 * t_round**2 / delta)
        return (8. * math.sqrt(self.dim * log_term_1 * log_term_2) \
            + 4. * math.sqrt(self.dim) * log_term_2 + math.sqrt(self.lambda_init) * self.args.norm_b +  2**ell * math.sqrt(self.dim)) 
    
    def calcuGamma(self, t_round, ell, delta = 0.033):
        log_term_1 = math.sqrt(math.log(1 + t_round * self.args.norm_a**2 / ((self.noise_range + 1)**2 * self.lambda_init)))
        log_term_2 = math.log(4 * t_round**2 * self.ell_max / delta)
        return 8. * math.sqrt(self.dim * log_term_1 * log_term_2) + 4. * math.sqrt(self.dim) * log_term_2 \
            + (math.log(2 * ell**2 / delta) + 3) * math.sqrt(self.dim) + math.sqrt(self.lambda_init) * self.args.norm_b
    
    
    def action(self, t_round, decision_set):
        MAX_R = float("-inf")

        ell_main = random.choices(self.levels, self.levels_weight)[0]

        self.ell_main = ell_main
        
        for a_t in range(self.num_actions):
            action_t = decision_set[a_t]
            r_action_t = min(np.dot(action_t, self.gl_mu) + self.C * self.calcuBeta(t_round, ell_main) * math.sqrt(np.mat(action_t) * np.mat(self.gl_Sigma).I * np.mat(action_t).T), 
                np.dot(action_t, self.sub_mu[ell_main]) + 1.0 * self.C * self.calcuGamma(t_round, ell_main) * math.sqrt(np.mat(action_t) * np.mat(self.sub_Sigma[self.ell_main]).I * np.mat(action_t).T))

            if MAX_R < r_action_t: 
                MAX_R = r_action_t
                final_a_t = a_t
                self.action_t = action_t
        
        return final_a_t
    
    def update(self, reward, sigma = 1.):
        gl_Sigma = np.mat(self.gl_Sigma)
        gl_Sigma = gl_Sigma + np.mat(self.action_t).T * np.mat(self.action_t) / sigma**2
        sub_Sigma = np.mat(self.sub_Sigma[self.ell_main])
        sub_Sigma = sub_Sigma + np.mat(self.action_t).T * np.mat(self.action_t) / sigma**2
        self.gl_bbb += reward * self.action_t / sigma**2
        self.sub_bbb[self.ell_main] += reward * self.action_t / sigma**2
        gl_mu = np.mat(self.gl_mu)
        sub_mu = np.mat(self.sub_mu[self.ell_main])
        gl_mu = (gl_Sigma.I * np.mat(self.gl_bbb).T).T
        sub_mu = (sub_Sigma.I * np.mat(self.sub_bbb[self.ell_main]).T).T
        self.gl_Sigma = gl_Sigma.A
        self.sub_Sigma[self.ell_main] = sub_Sigma
        self.gl_mu = gl_mu.A1
        self.sub_mu[self.ell_main] = sub_mu.A1

class OFUL(): 
    def __init__(self, args, lambda_init = 1., C = 0.05):
        self.ell_max = int(math.log(args.num_rounds * 2, 2) + 1)
        self.dim = args.hidden_dim
        self.sub_Sigma = np.zeros((self.ell_max + 1, self.dim, self.dim))
        for i in range(self.ell_max + 1):
            self.sub_Sigma[i] = np.identity(self.dim) * lambda_init
        self.gl_Sigma = np.identity(self.dim) * lambda_init
        self.gl_mu = np.zeros(self.dim)
        self.sub_mu = np.zeros((self.ell_max + 1, self.dim))
        self.gl_bbb = np.zeros(self.dim)
        self.sub_bbb = np.zeros((self.ell_max + 1, self.dim))
        self.noise_range = args.range
        self.crr = args.corruption_level
        self.num_rounds = args.num_rounds
        self.num_trials = args.num_trials
        self.num_actions = args.num_actions
        self.lambda_init = lambda_init
        self.args = args
        self.levels = [i for i in range(1, self.ell_max + 1)]
        self.levels_weight = [2**(-ell) for ell in range(1, self.ell_max + 1)]
        self.levels_weight[0] += 2**(-self.ell_max)
        self.C = C

    
    def calcuBeta(self, t_round, ell, delta = 0.1):
        log_term_1 = math.sqrt(math.log(1 + t_round * self.args.norm_a**2 / ((self.noise_range + 1)**2 * self.lambda_init)))
        log_term_2 = math.log(4 * t_round**2 / delta)
        return self.C * (8. * math.sqrt(self.dim * log_term_1 * log_term_2) \
            + 4. * math.sqrt(self.dim) * log_term_2 + math.sqrt(self.lambda_init) * self.args.norm_b)
    
    
    def action(self, t_round, decision_set):
        MAX_R = float("-inf")

        ell_main = random.choices(self.levels, self.levels_weight)[0]
        self.ell_main = ell_main
        
        for a_t in range(self.num_actions):
            action_t = decision_set[a_t]
            r_action_t = np.dot(action_t, self.gl_mu) + self.calcuBeta(t_round, ell_main) * math.sqrt(np.mat(action_t) * np.mat(self.gl_Sigma).I * np.mat(action_t).T)
            if MAX_R < r_action_t: 
                MAX_R = r_action_t
                final_a_t = a_t
                self.action_t = action_t
        
        return final_a_t
    
    def update(self, reward, sigma = 1.0):
        gl_Sigma = np.mat(self.gl_Sigma)
        gl_Sigma = gl_Sigma + np.mat(self.action_t).T * np.mat(self.action_t) / sigma**2
        sub_Sigma = np.mat(self.sub_Sigma[self.ell_main])
        sub_Sigma = sub_Sigma + np.mat(self.action_t).T * np.mat(self.action_t) / sigma**2
        self.gl_bbb += reward * self.action_t / sigma**2
        self.sub_bbb[self.ell_main] += reward * self.action_t / sigma**2
        gl_mu = np.mat(self.gl_mu)
        sub_mu = np.mat(self.sub_mu[self.ell_main])
        gl_mu = (gl_Sigma.I * np.mat(self.gl_bbb).T).T
        sub_mu = (sub_Sigma.I * np.mat(self.sub_bbb[self.ell_main]).T).T
        self.gl_Sigma = gl_Sigma.A
        self.sub_Sigma[self.ell_main] = sub_Sigma.A
        self.gl_mu = gl_mu.A1
        self.sub_mu[self.ell_main] = sub_mu.A1

class R_OFUL(): 
    def __init__(self, args, lambda_init = 1., C = 0.05):
        self.ell_max = int(math.log(args.num_rounds * 2, 2) + 1)
        self.dim = args.hidden_dim
        self.sub_Sigma = np.zeros((self.ell_max + 1, self.dim, self.dim))
        for i in range(self.ell_max + 1):
            self.sub_Sigma[i] = np.identity(self.dim) * lambda_init
        self.gl_Sigma = np.identity(self.dim) * lambda_init
        self.gl_mu = np.zeros(self.dim)
        self.sub_mu = np.zeros((self.ell_max + 1, self.dim))
        self.gl_bbb = np.zeros(self.dim)
        self.sub_bbb = np.zeros((self.ell_max + 1, self.dim))
        self.noise_range = args.range
        self.crr = args.corruption_level
        self.num_rounds = args.num_rounds
        self.num_trials = args.num_trials
        self.num_actions = args.num_actions
        self.lambda_init = lambda_init
        self.args = args
        self.levels = [i for i in range(1, self.ell_max + 1)]
        self.levels_weight = [2**(-ell) for ell in range(1, self.ell_max + 1)]
        self.levels_weight[0] += 2**(-self.ell_max)
        self.C = C

    
    def calcuBeta(self, t_round, ell, delta = 0.1):
        log_term_1 = math.sqrt(math.log(1 + t_round * self.args.norm_a**2 / ((self.noise_range + 1)**2 * self.lambda_init)))
        log_term_2 = math.log(4 * t_round**2 / delta)
        return self.C * (8. * math.sqrt(self.dim * log_term_1 * log_term_2) \
            + 4. * math.sqrt(self.dim) * log_term_2 + math.sqrt(self.lambda_init) * self.args.norm_b + 2. * self.crr * math.sqrt(self.dim))
    
    def action(self, t_round, decision_set):
        MAX_R = float("-inf")

        ell_main = random.choices(self.levels, self.levels_weight)[0]
        self.ell_main = ell_main
        
        for a_t in range(self.num_actions):
            action_t = decision_set[a_t]
            r_action_t = np.dot(action_t, self.gl_mu) + self.calcuBeta(t_round, ell_main) * math.sqrt(np.mat(action_t) * np.mat(self.gl_Sigma).I * np.mat(action_t).T)
            if MAX_R < r_action_t: 
                MAX_R = r_action_t
                final_a_t = a_t
                self.action_t = action_t
        
        return final_a_t
    
    def update(self, reward, sigma = 1.0):
        gl_Sigma = np.mat(self.gl_Sigma)
        gl_Sigma = gl_Sigma + np.mat(self.action_t).T * np.mat(self.action_t) / sigma**2
        sub_Sigma = np.mat(self.sub_Sigma[self.ell_main])
        sub_Sigma = sub_Sigma + np.mat(self.action_t).T * np.mat(self.action_t) / sigma**2
        self.gl_bbb += reward * self.action_t / sigma**2
        self.sub_bbb[self.ell_main] += reward * self.action_t / sigma**2
        gl_mu = np.mat(self.gl_mu)
        sub_mu = np.mat(self.sub_mu[self.ell_main])
        gl_mu = (gl_Sigma.I * np.mat(self.gl_bbb).T).T
        sub_mu = (sub_Sigma.I * np.mat(self.sub_bbb[self.ell_main]).T).T
        self.gl_Sigma = gl_Sigma.A
        self.sub_Sigma[self.ell_main] = sub_Sigma.A
        self.gl_mu = gl_mu.A1
        self.sub_mu[self.ell_main] = sub_mu.A1

class GREEDY(OFUL): 
    def calcuBeta(self, t_round, ell, delta = 0.1):
        return 0. 

class RobustB():
    def __init__(self, args, H, C, lambda_init = 1.0):
        self.dim = args.hidden_dim
        self.num_rounds = args.num_rounds
        self.noise_range = args.range
        self.num_actions = args.num_actions
        self.num_rounds = args.num_rounds
        self.JMax = math.ceil(math.log(args.num_rounds * args.num_actions * 2, 2))
        self.JSet = [0] + [2.**i for i in range(self.JMax + 1)]
        self.H = H
        self.C = C
        self.lambda_init = lambda_init
        self.prob_list = [1.] * len(self.JSet)
        self.alpha = min(1., math.sqrt(len(self.JSet) * math.log(len(self.JSet)) / ((math.e - 1) * math.ceil(self.num_rounds / self.H))))
        self.refresh()
    
    def refresh(self):
        self.Sigma = np.identity(self.dim) * self.lambda_init
        self.bbb = np.zeros(self.dim)
        self.mu = np.zeros(self.dim)
    
    def calcuBeta(self, t_round, layer, delta = 0.1):
        return self.C * (self.noise_range * math.sqrt(self.dim * math.log((1 + t_round / self.lambda_init) / delta)) + self.calcuGamma(t_round) * layer)
    
    def calcuGamma(self, t_round):
        return 2. * self.dim * math.log(1 + (t_round / self.dim * self.lambda_init))
    
    def action(self, t_round, decision_set):
        if t_round % self.H == 1:
            # self.refresh()
            self.probs = [(1 - self.alpha) * self.prob_list[i] / sum(self.prob_list) + self.alpha / len(self.JSet) for i in range(len(self.JSet))]
            self.layer_ind = np.random.choice(list(range(len(self.JSet))), p=self.probs)
            self.layer = self.JSet[self.layer_ind]
            # self.layer = 0. 
            # print(self.layer)
        MAX_R = float("-inf")
        t_round %= self.H
        for a_t in range(self.num_actions):
            action_t = decision_set[a_t]
            r_action_t = np.dot(self.mu, action_t) + self.calcuBeta(t_round, self.layer) * math.sqrt(np.mat(action_t) * np.mat(self.Sigma).I * np.mat(action_t).T)
            if r_action_t > MAX_R:
                MAX_R = r_action_t
                final_a_t = a_t
                self.action_t = action_t
        return final_a_t
    
    def update(self, reward):
        Sigma = np.mat(self.Sigma)
        Sigma = Sigma + np.mat(self.action_t).T * np.mat(self.action_t)
        self.Sigma = Sigma.A
        self.bbb = self.bbb + reward * self.action_t
        mu = (Sigma.I * np.mat(self.bbb).T).T
        self.mu = mu.A1
        self.prob_list[self.layer_ind] = self.prob_list[self.layer_ind] * math.exp(self.alpha * reward / (self.probs[self.layer_ind] * len(self.JSet)))


def experiment(ind, FILE_NAME, args, crr):
    dim = args.hidden_dim
    noise_range = args.range
    num_rounds = args.num_rounds
    num_actions = args.num_actions

    bmu = args.norm_a * np.ones(shape=(dim, )) / math.sqrt(dim)


    MO_learner = M_OFUL(args, C = 0.5)
    O_learner = OFUL(args, C = 0.0008)
    RO_learner = R_OFUL(args, C = .6)
    W_learner = OFUL(args, C = 0.002)
    G_learner = GREEDY(args)
    RB_learner = RobustB(args, H = 0.2 * int(math.sqrt(args.num_rounds)), C = .05)

    MO_regret = 0.
    O_regret = 0.
    RO_regret = 0.
    W_regret = 0.
    G_regret = 0.
    RB_regret = 0. 
    decision_t = init_arms(dim, args.norm_a, num_actions)

    MO_regret_list = []
    O_regret_list = []
    RO_regret_list = []
    W_regret_list = []
    G_regret_list = []
    RB_regret_list = []

    
    cur_crr = 0
    
    for t_round in range(1, num_rounds + 1):
        sigma = random.random() * 0.05
        flag = 0
        if cur_crr < crr:
            decision = init_arms(dim, args.norm_a, num_actions)
            if cur_crr < crr:
                flag = 1
        else:
            decision = decision_t
        noise = np.clip(np.random.randn(num_actions) * sigma, -noise_range, noise_range)
        rewardS = np.random.randn(num_actions)
        reward = np.random.randn(num_actions)
        optimal_reward = float("-inf")
        for arm in range(num_actions):
            rewardS[arm] = noise[arm] + np.dot(decision[arm], bmu)
            reward[arm] = rewardS[arm]
            if np.dot(decision[arm], bmu) > optimal_reward:
                optimal_reward = np.dot(decision[arm], bmu)
        
        if flag == 1:
            cur_crr += 1
            for arm in range(num_actions):
                reward[arm] = - np.dot(decision[arm], bmu)
        
        
        action_t_id = MO_learner.action(t_round, decision)
        action_t = decision[action_t_id]
        MO_regret += optimal_reward - np.dot(action_t, bmu)
        MO_learner.update(reward[action_t_id], sigma = sigma)

        action_t_id = O_learner.action(t_round, decision)
        action_t = decision[action_t_id]
        O_regret += optimal_reward - np.dot(action_t, bmu)
        O_learner.update(reward[action_t_id])

        action_t_id = RO_learner.action(t_round, decision)
        action_t = decision[action_t_id]
        RO_regret += optimal_reward - np.dot(action_t, bmu)
        RO_learner.update(reward[action_t_id], sigma = sigma)

        action_t_id = W_learner.action(t_round, decision)
        action_t = decision[action_t_id]
        W_regret += optimal_reward - np.dot(action_t, bmu)
        W_learner.update(reward[action_t_id], sigma = sigma)

        action_t_id = G_learner.action(t_round, decision)
        action_t = decision[action_t_id]
        G_regret += optimal_reward - np.dot(action_t, bmu)
        G_learner.update(reward[action_t_id])

        action_t_id = RB_learner.action(t_round, decision)
        action_t = decision[action_t_id]
        RB_regret += optimal_reward - np.dot(action_t, bmu)
        RB_learner.update(reward[action_t_id])

        if t_round % 100 == 0:
            MO_regret_list.append(MO_regret)
            O_regret_list.append(O_regret)
            RO_regret_list.append(RO_regret)
            W_regret_list.append(W_regret)
            G_regret_list.append(G_regret)
            RB_regret_list.append(RB_regret)
            print("Experiment " + str(ind) + " crr " + str(crr) + " round " + str(t_round))
            print("Multi-level " + str(MO_regret))
            print("OFUL " + str(O_regret))
            print("R-OFUL " + str(RO_regret))
            print("Weighted " + str(W_regret))
            print("GREEDY " + str(G_regret))
            print("RobustB " + str(RB_regret))
    
    data = dict()
    data['Multi-level-OFUL'] = MO_regret_list
    data['Weighted-OFUL'] = W_regret_list
    data['OFUL'] = O_regret_list
    data['R-OFUL'] = RO_regret_list
    data['GREEDY'] = G_regret_list
    data['RobustBandit'] = RB_regret_list

    with open(FILE_NAME + '_{}.txt'.format(str(ind)), 'w') as outfile:
        json.dump(data, outfile)

if __name__ == "__main__":
    args = get_args()
    dim = args.hidden_dim
    noise_range = args.range
    crr = args.corruption_level
    num_rounds = args.num_rounds
    num_trials = args.num_trials
    
    num_actions = args.num_actions

    random.seed(args.seed)
    np.random.seed(args.seed)

    levels = [450, 300, 150, 90, 30, 0]
    
    for i in range(num_trials):
        
        for j in levels:
            crr = j
            experiment(i, 'test_crr_' + str(crr), args, crr)

