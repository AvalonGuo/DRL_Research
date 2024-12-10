# Policy Optimization
## Policy Gradient Methods
Policy gradient methods work by __computing an estimator of the policy gradient(近似估计)__ and plugging it into a stochastic gradient ascent algorithm.The most commonly used gradient estimator has the form:
```math
\hat{g} = \widehat{\mathbb{E}}_t \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \hat{A}_t \right]
```
* $\hat{g}$ represents the estimated gradient of the expected return.
* $\hat{\mathbb{E}_t }$ denotes the empirical expectation over time steps t.
* $\nabla_\theta$ is the gradient with respect to the parameters θ of the policy.
* $\log \pi_\theta(a_t \mid s_t)$  is the log-probability of taking action $a_t$ given state $s_t$ under the policy $\pi_\theta$.
* $\hat{A}_t$ is an estimate of the advantage function at time step t, which measures how much better it was to take action $a_t$ in state $s_t$ compared to what might be expected under the current policy.

In essence, this formula tells us how to update the policy parameters 
θ to increase the probability of actions that led to higher rewards, based on the estimated advantage of those actions. Implementations that use automatic differentiation software work by constructing an objective function(构造代理函数) whose gradient is the policy gradient estimator. The estimator $\hat{g}$ is obtained by differentiating the objective:
```math
L^{PG}(\theta) = \hat{\mathbb{E}}_t \left[ \log \pi_\theta(a_t \mid s_t) \hat{A}_t \right]
```

## Trust Region Methods
```math
\begin{aligned}
& \text{maximize} & & \hat{\mathbb{E}}_t \left[ \frac{\pi_{\theta}(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} \hat{A}_t \right] & \text{subject to} & & \hat{\mathbb{E}}_t [\text{KL}[\pi_{\theta_{\text{old}}}(\cdot \mid s_t), \pi_{\theta}(\cdot \mid s_t)]] \leq \delta.
\end{aligned}
```
