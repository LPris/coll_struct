import numpy as np


class Struct:

    def __init__(self, config=None):
        if config is None:
            config = {"n_comp": 22,
                      "discount_reward": 0.95,
                      "campaign_cost": False}
        assert "n_comp" in config and \
               "discount_reward" in config and \
               "campaign_cost" in config, \
            "Missing env config"

        self.n_comp = config["n_comp"]
        self.discount_reward = config["discount_reward"]
        self.campaign_cost = config["campaign_cost"]
        self.time = 0
        self.ep_length = 30  # Horizon length
        self.n_st_comp = 30  # Crack states (fatigue hotspot damage states)
        self.n_obs = 2
        # Total number of observations per hotspot (crack detected / crack not detected)
        self.actions_per_agent = 3

        # Uncorrelated obs = 30 per agent + 1 timestep
        # Correlated obs = 30 per agent + 1 timestep +
        #                   80 hyperparameter states = 111
        self.obs_per_agent_multi = None  # Todo: check
        self.obs_total_single = None  # Todo: check used in gym env

        ### Loading the underlying POMDP model ###
        drmodel = np.load('pomdp_models/zayas_input.npz')

        # (ncomp components, nstcomp crack states)
        self.belief0 = np.zeros((self.n_comp, self.n_st_comp))

        self.indZayas = drmodel['indZayas']
        self.relSysc = drmodel['relSysc']
        self.comp_agent = drmodel['comp_agent'] #Link between element agent and component
        self.n_elem = 13

        self.belief0[:, :] = drmodel['belief0'][0, 0, :, 0]

        # (3 actions, 10 components, 31 det rates, 30 cracks, 30 cracks)
        self.P = drmodel['P'][:, 0, :, :, :]

        # (3 actions, 10 components, 30 cracks, 2 observations)
        self.O = drmodel['O'][:, 0, :, :]

        self.agent_list = ["agent_" + str(i) for i in range(self.n_elem)] # one agent per element

        self.time_step = 0
        self.beliefs = self.belief0
        self.d_rate = np.zeros((self.n_comp, 1), dtype=int)
        self.observations = None

        # Reset struct_env.
        self.reset()

    def reset(self):
        # We need the following line to seed self.np_random
        # super().reset(seed=seed)

        # Choose the agent's belief
        self.time_step = 0
        self.beliefs = self.belief0
        self.d_rate = np.zeros((self.n_comp, 1), dtype=int)
        self.observations = {}
        for i in range(self.n_elem):
            beliefs_agents = self.beliefs[self.indZayas[i,0]:self.indZayas[i,1] + 1] if self.indZayas[i,1]>0 else self.beliefs[self.indZayas[i,0]]
            self.observations[self.agent_list[i]] = np.concatenate(
                (beliefs_agents.reshape(-1), [self.time_step / self.ep_length]))

        return self.observations

    def step(self, action: dict):
        action_ = np.zeros(self.n_comp, dtype=int)
        for i in range(self.n_comp):
            action_[i] = action[self.agent_list[i]]

        observation_, belief_prime, drate_prime = \
            self.belief_update_uncorrelated(self.beliefs, action_,
                                            self.d_rate)



        reward_ = self.immediate_cost(self.beliefs, action_, belief_prime,
                                      self.d_rate)
        reward = self.discount_reward ** self.time_step * reward_.item()  # Convert float64 to float

        rewards = {}
        for i in range(self.n_comp):
            rewards[self.agent_list[i]] = reward

        self.time_step += 1

        self.observations = {}
        for i in range(self.n_comp):
            self.observations[self.agent_list[i]] = np.concatenate(
                (belief_prime[i], [self.time_step / self.ep_length]))

        self.beliefs = belief_prime
        self.d_rate = drate_prime

        # An episode is done if the agent has reached the target
        done = self.time_step >= self.ep_length

        # info = {"belief": self.beliefs}
        return self.observations, rewards, done

    def connectZayas(self, pf, indZayas): # from component state to element state # (Add here brace failure prob)!!
        relComp = 1 - pf
        relComp = np.append(relComp, 1)
        relEl = np.zeros(self.n_elem)
        for i in range(self.n_elem):
            relEl[i] = relComp[ indZayas[i,0] ] * relComp[ indZayas[i,1] ]
        return relEl

    def elemState(self, pfEl, nEl): # from element state to element event #
        qcomp = np.array([pfEl[-1], 1 - pfEl[-1]]) # first component
        qprev = qcomp.copy() # initialize iterative procedure
        for j in range(nEl - 1):
            qnew = np.repeat( np.array([pfEl[-2-j], 1 - pfEl[-2-j]]), qprev.shape )
            qprev = np.tile(qprev, 2)
            qc = np.multiply(qprev, qnew)
            qprev = qc
        return qc

    def pf_sys(self, pf): # system failure probability #
        pfEl = 1 - self.connectZayas(pf, self.indZayas) 
        q = self.elemState(pfEl, self.n_elem)
        rel_ = self.relSysc.T.dot(q)
        PF_sys = 1 - rel_
        return PF_sys

    def immediate_cost(self, B, a, B_, drate):
        """ immediate reward (-cost),
         based on current damage state and action """
        cost_system = 0
        PF = B[:, -1]
        PF_ = B_[:, -1].copy()
        campaign_executed = False
        for i in range(self.n_comp):
            if a[i] == 1:
                cost_system += -0.2 if self.campaign_cost else -1 # Individual inspection costs 
                Bplus = self.P[a[i], drate[i, 0]].T.dot(B[i, :])
                PF_[i] = Bplus[-1]
                if self.campaign_cost and not campaign_executed:
                    campaign_executed = True # Campaign executed
            elif a[i] == 2:
                cost_system += - 15
                if self.campaign_cost and not campaign_executed:
                    campaign_executed = True # Campaign executed
        if self.n_comp < 2:  # single component setting
            PfSyS_ = PF_
            PfSyS = PF
        else:
            PfSyS_ = self.pf_sys(PF_, self.k_comp)
            PfSyS = self.pf_sys(PF, self.k_comp)
        if PfSyS_ < PfSyS:
            cost_system += PfSyS_ * (-50000)
        else:
            cost_system += (PfSyS_ - PfSyS) * (-50000)
        if campaign_executed: # Assign campaign cost
            cost_system += -5
        return cost_system

    def belief_update_uncorrelated(self, b, a, drate):
        """Bayesian belief update based on
         previous belief, current observation, and action taken"""
        b_prime = np.zeros((self.n_comp, self.n_st_comp))
        b_prime[:] = b
        ob = np.zeros(self.n_comp)
        drate_prime = np.zeros((self.n_comp, 1), dtype=int)
        for i in range(self.n_comp):
            p1 = self.P[a[i], drate[i, 0]].T.dot(
                b_prime[i, :])  # environment transition

            b_prime[i, :] = p1
            # if do nothing, you update your belief without new evidences
            drate_prime[i, 0] = drate[i, 0] + 1
            # At every timestep, the deterioration rate increases

            ob[i] = 2  # ib[o] = 0 if no crack detected 1 if crack detected
            if a[i] == 1:
                Obs0 = np.sum(p1 * self.O[a[i], :, 0])
                # self.O = Probability to observe the crack
                Obs1 = 1 - Obs0

                if Obs1 < 1e-5:
                    ob[i] = 0
                else:
                    ob_dist = np.array([Obs0, Obs1])
                    ob[i] = np.random.choice(range(0, self.n_obs), size=None,
                                             replace=True, p=ob_dist)
                b_prime[i, :] = p1 * self.O[a[i], :, int(ob[i])] / (
                    p1.dot(self.O[a[i], :, int(ob[i])]))  # belief update
            if a[i] == 2:
                # action in b_prime has already
                # been accounted in the env transition
                drate_prime[i, 0] = 0
        return ob, b_prime, drate_prime
