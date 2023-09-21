## Reinforce

The repo provide the implementation of Reinforce.

Pay attention! the update process of Reinforce is done after the trajectory end.

The loss function covers the policy gradient, we can get it by computing the derivative of the objective of RL.

$J_\pi = - G_t * \log \pi(a|s)$

Tips:
1. You can use `torch.distribution.Categorical` to model the action distribution.
2. You can run the code with `python main_reinforce.py`.

We use `gynasium-Cartpole-v1` to test the discrete variant.

![The training curve on gynasium-Cartpole-v1](Gymnasium-CartPole-v1.png)