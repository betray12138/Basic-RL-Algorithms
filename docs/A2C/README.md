## A2C (Advantage Actor-Critic)

The repo provide two variants of A2C, namely A2CContinuous and A2CDiscrete.

The core of implementing A2C is update per step to improve the sample efficiency compared to `REINFORCE`.

$J_\pi = - (r + \gamma V(s') - V(s)) * \log \pi(a|s)$

Tips:

1. When updating the policy network, td-target is of no gradient, so you should use `(r + γ*v_s_ - v_s).detach()`.
2. For discrete variant, you can use `torch.distribution.Categorical` to model the action distribution.
3. You can run the code with `python main_a2c_continuous.py` or `python main_a2c_discrete.py`.

We use `gymnasium-Pendulum-v1` to test the continuous variant and use `gynasium-Cartpole-v1` to test the discrete variant.

![The training curve on gymnasium-Pendulum-v1](Gymnasium-Pendulum-v1.png)

![The training curve on gynasium-Cartpole-v1](Gymnasium-CartPole-v1.png)