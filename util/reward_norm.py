import numpy as np

# 根据Welford公式完成在线方差和标准差的更新
class ZeroNormalization(object):
    def __init__(self, dim: int) -> None:
        super(ZeroNormalization, self).__init__()
        self.dim = dim
        self.total = 0
        self.accumulate_sum_var = np.zeros(self.dim)
        self.mean = np.zeros(self.dim)
    
    def update(self, x): 
        self.total += 1
        delta = x - self.mean
        # 更新mean
        self.mean = self.mean + delta / self.total
        
        # 更新方差的累计和
        delta_ = x - self.mean
        self.accumulate_sum_var = self.accumulate_sum_var + delta * delta_
    
    def get_mean_std(self):
        return self.mean, np.sqrt(self.accumulate_sum_var / self.total)
    

# 动态计算折扣奖励累积和的标准差，且对当前reward除以该标准差
class RewardScaling(object):
    def __init__(self, gamma) -> None:
        self.norm_maintainer = ZeroNormalization(1)
        self.gamma = gamma
        self.accumulated_discount_reward = 0
        
    def get_and_update(self, reward):
        self.accumulated_discount_reward = self.gamma * self.accumulated_discount_reward + reward
        self.norm_maintainer.update(self.accumulated_discount_reward)
        _, std = self.norm_maintainer.get_mean_std()
        return reward / (std + 1e-6)    # +1e-6防止除0

    def reset(self):
        # 每一次轨迹开始都需要重置accumulated_discount_reward
        self.accumulated_discount_reward = 0