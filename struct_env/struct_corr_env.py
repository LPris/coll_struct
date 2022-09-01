import numpy as np


class Struct:

    def __init__(self, config=None):
        if config is None:
            config = {"components": 2, "discount_reward": 1, "k_comp": None}
        self.ncomp = config["components"]
        self.discount_reward = config["discount_reward"]
        self.k_comp = self.ncomp-1 if config["k_comp"] is None else config["k_comp"]
        self.time = 0
        self.ep_length = 30
        self.nstcomp = 30  # What is this?
        self.nsthyperp = 80
        self.nobs = 2  # What is this?
        self.actions_per_agent = 3

        # obs = 30 per agent + 1 timestep + 80 hyperparameter states = 111
        self.obs_per_agent_multi = 111
        self.obs_total_single = 30 * self.ncomp + 1 + 80

        ### Loading the underlying POMDP model ###
        drmodel = np.load('pomdp_models/Dr3031_H08.npz')

        # (n components, 30 crack states)
        self.belief0 = np.zeros((self.ncomp, self.nstcomp))
        self.belief0[:, :] = drmodel['belief0']

        # conditional beliefs (n components, 80 hyperparameter states, 30 crack states)
        self.belief0c = np.zeros((self.ncomp, self.nsthyperp, self.nstcomp))
        self.belief0c[:, :, :] = drmodel['belief0c']

        # conditional beliefs associated with a repair action (80 hyperparameter states, 30 crack states)
        self.b0cR = drmodel['b0cR']

        # hyperparameter marginal states
        self.alpha0 = drmodel['alpha0']

        # (3 actions, 31 det rates, 30 cracks, 30 cracks)
        self.P = drmodel['P']

        # (3 actions, 30 cracks, 2 observations)
        self.O = drmodel['O']

        self.agent_list = ["agent_" + str(i) for i in range(self.ncomp)]

        self.time_step = 0
        self.beliefs = self.belief0
        self.beliefsc = self.belief0c
        self.alphas = self.alpha0
        self.drate = np.zeros((self.ncomp, 1), dtype=int)
        self.observations = None

        # Reset struct_env.
        self.reset()

    def reset(self):
        # We need the following line to seed self.np_random
        # super().reset(seed=seed)

        # Choose the agent's belief
        self.time_step = 0
        self.beliefs = self.belief0
        self.beliefsc = self.belief0c
        self.alphas = self.alpha0
        self.drate = np.zeros((self.ncomp, 1), dtype=int)
        self.observations = {}
        for i in range(self.ncomp):
            self.observations[self.agent_list[i]] = np.concatenate(
                (self.beliefs[i], self.alphas, [self.time_step / 30]))
        return self.observations

    def step(self, action: dict):
        action_ = np.zeros(self.ncomp, dtype=int)
        for i in range(self.ncomp):
            action_[i] = action[self.agent_list[i]]

        observation_, belief_prime, drate_prime, bc_prime, alpha_prime = self.belief_update(
            self.beliefsc, action_, self.drate, self.alphas)

        reward_ = self.immediate_cost(self.beliefs, action_, belief_prime,
                                      self.drate)
        reward = self.discount_reward ** self.time_step * reward_.item()  # Convert float64 to float

        rewards = {}
        for i in range(self.ncomp):
            rewards[self.agent_list[i]] = reward

        self.time_step += 1

        self.observations = {}
        for i in range(self.ncomp):
            self.observations[self.agent_list[i]] = np.concatenate(
                (belief_prime[i], alpha_prime, [self.time_step / 30]))

        self.beliefs = belief_prime
        self.beliefsc = bc_prime
        self.alphas = alpha_prime
        self.drate = drate_prime

        # An episode is done if the agent has reached the target
        done = self.time_step >= self.ep_length

        # info = {"belief": self.beliefs}
        return self.observations, rewards, done

    def pf_sys(self, pf, k):  # compute pf_sys for k-out-of-n components
        n = pf.size
        # k = ncomp-1
        nk = n - k
        m = k + 1
        A = np.zeros(m + 1)
        A[1] = 1
        L = 1
        for j in range(1, n + 1):
            h = j + 1
            Rel = 1 - pf[j - 1]
            if nk < j:
                L = h - nk
            if k < j:
                A[m] = A[m] + A[k] * Rel
                h = k
            for i in range(h, L - 1, -1):
                A[i] = A[i] + (A[i - 1] - A[i]) * Rel
        PF_sys = 1 - A[m]
        return PF_sys

    def immediate_cost(self, B, a, B_,
                       drate):  # immediate reward (-cost), based on current damage state and action#
        cost_system = 0
        PF = B[:, -1]
        PF_ = B_[:, -1].copy()

        for i in range(self.ncomp):
            if a[i] == 1:
                cost_system += -1
                Bplus = self.P[a[i], drate[i, 0]].T.dot(B[i, :])
                PF_[i] = Bplus[-1]
            elif a[i] == 2:
                cost_system += - 20
        if self.ncomp < 2:  # single component setting
            PfSyS_ = PF_
            PfSyS = PF
        else:
            PfSyS_ = self.pf_sys(PF_, self.k_comp)
            PfSyS = self.pf_sys(PF, self.k_comp)
        if PfSyS_ < PfSyS:
            cost_system += PfSyS_ * (-10000)
        else:
            cost_system += (PfSyS_ - PfSyS) * (-10000)
        return cost_system

    def belief_update(self, bc, a,
                      drate, alpha):  # Bayesian belief update based on previous belief, current observation, and action taken
        b_prime = np.zeros((self.ncomp, self.nstcomp))
        bc_prime = np.zeros((self.ncomp, self.nsthyperp, self.nstcomp))
        drate_prime = np.zeros((self.ncomp, 1), dtype=int)
        alpha_prime = np.zeros((self.nsthyperp))
        alpha_prime = alpha.copy()
        ob = np.zeros(self.ncomp)
        for i in range(self.ncomp):
            p1 = bc[i,:,:].dot(self.P[a[i], drate[i,0]])  # environment transition  
            bc_prime[i,:,:] = p1
            drate_prime[i, 0] = drate[i, 0] + 1

            if a[i] == 1:
                Obs0 = np.sum(alpha[:].dot(p1)* self.O[a[i],:,0])
                Obs1 = 1 - Obs0
                if Obs1 < 1e-5:
                    ob[i] = 0
                else:
                    ob_dist = np.array([Obs0, Obs1])
                    ob[i] = np.random.choice(range(0,self.nobs), size=None, replace=True, p=ob_dist)   
                pInsp = p1* self.O[a[i], :, int(ob[i])]  # belief update
                likAlpha = np.sum(pInsp, axis=1)  # likelihood insp alpha
                normBel = np.tile(likAlpha, self.nstcomp).reshape(self.nsthyperp, self.nstcomp, order='F') # normalization constant
                bc_prime[i,:,:] = pInsp / normBel
                alpha_curr = likAlpha*alpha_prime
                alpha_prime = alpha_curr / np.sum(alpha_curr)

            if a[i] == 2:
                bc_prime[i,:,:] = self.b0cR
                drate_prime[i, 0] = 0

        for i in range(self.ncomp):
            b_prime[i,:] = alpha_prime.dot(bc_prime[i,:,:]) # Belief (marginalize out alpha)
        return ob, b_prime, drate_prime, bc_prime, alpha_prime
