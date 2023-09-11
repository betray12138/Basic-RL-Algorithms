import sys
import numpy as np

if 'gym' in sys.modules:
    # 此处完成对gym的封装即可
    import gym
    pass
elif 'gymnasium' in sys.modules:
    import gymnasium as gym


class GymWrapper(object):
    def __init__(self, env: gym.Env):
        super(GymWrapper, self).__init__()
        self.env = env
    
    
class GymNasiumWrapper(object):
    def __init__(self, env: gym.Env):
        super(GymNasiumWrapper, self).__init__()
        self.env = env
    
    def reset(self):
        state, _ = self.env.reset()
        return state
    
    def step(self, action: np.ndarray):
        # 返回s' r done truncated
        action = action.flatten()
        next_state, reward, done, truncated, _ = self.env.step(action)
        return next_state, reward, done, truncated

    def __getattr__(self, attr):
        return getattr(self.env, attr)
    
class GymNasiumDiscreteWrapper(object):
    def __init__(self, env: gym.Env):
        super(GymNasiumDiscreteWrapper, self).__init__()
        self.env = env
    
    def reset(self):
        state, _ = self.env.reset()
        return state
    
    def step(self, action: np.ndarray):
        # 返回s' r done truncated
        next_state, reward, done, truncated, _ = self.env.step(action[0])
        return next_state, reward, done, truncated

    def __getattr__(self, attr):
        return getattr(self.env, attr)