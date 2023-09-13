
# low -> high via last_timesteps
class LinearSchedule(object):
    def __init__(self, last_timesteps: int, low_val: float, high_val: float) -> None:
        self.last_tiemsteps = last_timesteps
        self.low_val = low_val
        self.high_val = high_val
    
    def get_value(self, cur_step: int):
        return min(self.high_val, self.low_val + cur_step * (self.high_val - self.low_val) / self.last_tiemsteps) 