import numpy as np

class Zayas:

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

        # Loading collision input information
        col_inputs = np.load('collisions/collision_info.npz')
        self.freq_col = col_inputs['freq_col']
        # self.freq_col = np.zeros(5) # This should be user defined #
        self.col_intens = col_inputs['col_intens']
        self.energy_max = col_inputs['energy_max']
        self.energy_max_index = col_inputs['energy_max_index']
        self.energy_max_samp = col_inputs['energy_max_samp']
        self.energy_impact = np.zeros(self.n_elem)
        self.pf_brace = np.zeros(self.n_elem)
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
        self.energy_impact = np.zeros(self.n_elem)
        self.pf_brace = np.zeros(self.n_elem)
        self.observations = {}

        for i in range(self.n_elem):
            if self.indZayas[i,1] > 0:
                beliefs_agents = self.beliefs[self.indZayas[i,0]:self.indZayas[i,1] + 1]  
            else:
                beliefs_agents = np.concatenate( ( [self.beliefs[self.indZayas[i,0]]], [np.zeros(30)] ))
            self.observations[self.agent_list[i]] = np.concatenate(
                (beliefs_agents.reshape(-1), [self.pf_brace[i]], [self.time_step / self.ep_length]))

        return self.observations

    def step(self, action: dict):
        action_ = np.zeros(self.n_elem, dtype=int)
        for i in range(self.n_elem):
            action_[i] = action[self.agent_list[i]]

        observation_, belief_prime, drate_prime = \
            self.belief_update_uncorrelated(self.beliefs, action_,
                                            self.d_rate)

        # energy_impact_prime = self.energy_collision(self.energy_impact)
        # pf_brace_prime = self.pf_col(energy_impact_prime)
        # print(energy_impact_prime, pf_brace_prime)

        reward_ = self.immediate_cost(self.beliefs, action_, belief_prime,
                                      self.d_rate)
        reward = self.discount_reward ** self.time_step * reward_.item()  # Convert float64 to float

        rewards = {}
        for i in range(self.n_elem):
            rewards[self.agent_list[i]] = reward

        self.time_step += 1

        self.observations = {}
        for i in range(self.n_elem):
            if self.indZayas[i,1] > 0:
                beliefs_agents = belief_prime[self.indZayas[i,0]:self.indZayas[i,1] + 1]  
            else:
                beliefs_agents = np.concatenate( ( [belief_prime[self.indZayas[i,0]]], [np.zeros(30)] ))
            self.observations[self.agent_list[i]] = np.concatenate(
                (beliefs_agents.reshape(-1), [self.pf_brace[i]], [self.time_step / self.ep_length]))

        self.beliefs = belief_prime
        self.d_rate = drate_prime

        # An episode is done if the agent has reached the target
        done = self.time_step >= self.ep_length

        # info = {"belief": self.beliefs}
        return self.observations, rewards, done, observation_

    def connectZayas(self, pf, indZayas, pf_brace): # from component state to element state # (Add here brace failure prob)!!
        relComp = 1 - pf
        relComp = np.append(relComp, 1)
        rel_brace = 1 - pf_brace
        relEl = np.zeros(self.n_elem)
        for i in range(self.n_elem):
            relEl[i] = relComp[ indZayas[i,0] ] * relComp[ indZayas[i,1] ] * rel_brace[i]
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

    def pf_sys(self, pf, pf_brace): # system failure probability #
        pfEl = 1 - self.connectZayas(pf, self.indZayas, pf_brace) 
        q = self.elemState(pfEl, self.n_elem)
        rel_ = self.relSysc.T.dot(q)
        PF_sys = 1 - rel_
        return PF_sys

    def immediate_cost(self, B, a, B_, drate):
        """ immediate reward (-cost),
         based on current damage state and action """
        cost_system = 0
        # hotspots pf 
        PF = B[:, -1]
        PF_ = B_[:, -1].copy()
        # brace collision pf
        campaign_executed = False
        pf_brace = self.pf_brace
        pf_brace_ = self.pf_brace.copy()
        energy_impact_prime = self.energy_collision(self.energy_impact)
        pf_brace_ = self.pf_col(energy_impact_prime)
        for i in range(self.n_elem):
            if a[i] == 1:
                cost_system += -0.4 if self.campaign_cost else -2 # Individual inspection costs 
                Bplus = self.P[a[i], drate[self.indZayas[i,0], 0]].T.dot(B[self.indZayas[i,0], :])
                PF_[self.indZayas[i,0]] = Bplus[-1]
                if self.indZayas[i,1] > 0:
                    Bplus = self.P[a[i], drate[self.indZayas[i,1], 0]].T.dot(B[self.indZayas[i,1], :])
                    PF_[self.indZayas[i,1]] = Bplus[-1]
                if self.campaign_cost and not campaign_executed:
                    campaign_executed = True # Campaign executed
            elif a[i] == 2:
                cost_system += - 30
                self.energy_impact[i] = 0
                pf_brace_[i] = 0
                if self.campaign_cost and not campaign_executed:
                    campaign_executed = True # Campaign executed
        
        PfSyS_ = self.pf_sys(PF_, pf_brace_)
        PfSyS = self.pf_sys(PF, pf_brace)
        self.pf_brace = pf_brace_
        if PfSyS_ < PfSyS:
            cost_system += PfSyS_ * (-50000)
        else:
            cost_system += (PfSyS_ - PfSyS) * (-50000)
        if campaign_executed: # Assign campaign cost
            cost_system += -5
        print(PfSyS, PfSyS_, cost_system)
        return cost_system

    def belief_update_uncorrelated(self, b, a, drate):
        """Bayesian belief update based on
         previous belief, current observation, and action taken"""
        b_prime = np.zeros((self.n_comp, self.n_st_comp))
        b_prime[:] = b
        ob = np.zeros(self.n_comp)
        drate_prime = np.zeros((self.n_comp, 1), dtype=int)
        for i in range(self.n_comp):
            p1 = self.P[a[self.comp_agent[i]], drate[i, 0]].T.dot(
                b_prime[i, :])  # environment transition

            b_prime[i, :] = p1
            # if do nothing, you update your belief without new evidences
            drate_prime[i, 0] = drate[i, 0] + 1
            # At every timestep, the deterioration rate increases

            ob[i] = 2  # ib[o] = 0 if no crack detected 1 if crack detected
            if a[self.comp_agent[i]] == 1:
                Obs0 = np.sum(p1 * self.O[a[self.comp_agent[i]], :, 0])
                # self.O = Probability to observe the crack
                Obs1 = 1 - Obs0

                if Obs1 < 1e-5:
                    ob[i] = 0
                else:
                    ob_dist = np.array([Obs0, Obs1])
                    ob[i] = np.random.choice(range(0, self.n_obs), size=None,
                                             replace=True, p=ob_dist)
                b_prime[i, :] = p1 * self.O[a[self.comp_agent[i]], :, int(ob[i])] / (
                    p1.dot(self.O[a[self.comp_agent[i]], :, int(ob[i])]))  # belief update
            if a[self.comp_agent[i]] == 2:
                # action in b_prime has already
                # been accounted in the env transition
                drate_prime[i, 0] = 0
        return ob, b_prime, drate_prime

    def energy_collision(self, impact_energy_start):
        collision_events = np.random.poisson(lam = self.freq_col, size=None)
        index_impact_braces = np.array([0, 1, 2, 3, 8, 9, 10, 11], dtype = int)
        impact_energy = impact_energy_start.copy()
        for i in range(5):
            if collision_events[i] > 0:
                for _ in range(collision_events[i]):
                    ind_energy = np.random.choice(index_impact_braces)
                    impact_energy[ind_energy] += self.col_intens[i]
        self.energy_impact = impact_energy
        return impact_energy
    
    def pf_col(self, energy):
        pf_brace = np.zeros(self.n_elem)
        for i in range(self.n_elem):
            if energy[i] > 0:
                pf_brace[i] = np.sum(self.energy_max[self.energy_max_index[i]] < energy[i]) / self.energy_max_samp
        return pf_brace
