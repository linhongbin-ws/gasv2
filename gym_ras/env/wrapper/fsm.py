from gym_ras.env.wrapper.base import BaseWrapper
import gym


class FSM(BaseWrapper):
    """ finite state machine 
    """

    def __init__(self, env,
                 states=[
                     "prog_norm",  # normal in progress
                     "prog_abnorm_1",  # abnormal 1 : hit workspace limit
                     "prog_abnorm_2",  # abnormal 2 : object sliding
                     "prog_abnorm_3",  # abnormal 3 : gripper toggle
                     "done_fail",  # done, failure case
                     "done_success",  # done success case
                 ],
                 **kwargs
                 ):
        super().__init__(env)
        self._states = states
        self._rewards = {}
        for s in self._states:
            self._rewards[s] = kwargs["reward_"+ s]
    
    def _get_state_data(self, state_name):
        obs_state = self._states.index(state_name)
        done =  state_name in ["done_success", "done_fail"]
        reward = self._rewards[state_name]
        return obs_state, reward, done
    

    def reset(self):
        obs = self.env.reset()
        state_name = "prog_norm"
        obs_state, _, _ = self._get_state_data(state_name)
        obs['fsm_state'] = obs_state
        return obs

    def step(self, action):
        obs, _, _, info = self.env.step(action)
        obs_state, reward, done = self._get_state_data(info['fsm'])
        obs['fsm_state'] = obs_state
        return obs, reward, done, info

    @property
    def observation_space(self):
        obs = {k: v for k, v in self.env.observation_space.items()}
        obs['fsm_state'] = gym.spaces.Box(low=0,
                                          high=len(self._states)-1, shape=(1,), dtype=float)
        return gym.spaces.Dict(obs)
    # @property
    # def fsm_states(self):
    #     return self._states