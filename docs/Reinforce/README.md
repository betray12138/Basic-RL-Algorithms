## Reinforce

The repo provide the implementation of Reinforce.

Pay attention! the update process of Reinforce is done after the trajectory end.

$J_\pi = - G_t * \log \pi(a|s)$

Tips:

1. When updating the policy network, td-target is of no gradient, so you should use `(r + γ*v_s_ - v_s).detach()`.
2. For discrete variant, you can use `torch.distribution.Categorical` to model the action distribution.
3. You can run the code with `python main_reinforce.py`.

We use `gynasium-Cartpole-v1` to test the discrete variant.

![The training curve on gynasium-Cartpole-v1](Gymnasium-CartPole-v1.png)