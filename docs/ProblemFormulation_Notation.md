# Introduction to Policy Optimization 

Reference: https://spinningup.openai.**com**/en/latest/spinningup/rl_intro3.html

## I. Notations:
### 1. Policy:
A policy is a rule used by an agent to decide what actions to take:
+ It can be deterministic, denoted as: $a_t=\mu(s_t)$
+ It may be stochastic, often denoted as: $a_t \sim \pi(\cdot | s_t)$

In Deep RL, a policy is parameterized by an Neural Network parameters $\theta$, hence we write the policy as:
+ deterministic: $a_t=\mu_{\theta}(s_t)$
+ stochastic: $a_t \sim \pi_{\theta}(\cdot | s_t)$. 
  
### 2. Trajectory: 
A trajectory $\tau$ is a sequence of states $s_t$ and actions $a_t$ in the world: $\tau=(s_0,a_0,s_1,a_1)$:
+ state $s_0$ is random assigned when starting env: $s_0 \sim \rho_0$.
+ state $s_t$ depends only on the current state and action, assuming that it is stochastic with transition probability: $s_t \sim P(\cdot|s_t,a_t)$. 
+ assuming action $a_t$ is ruled by the stochastic policy: $a_t \sim \pi_\theta(\cdot|s_t)$.
  
Then, the probability of T-step trajectory is:
    $$P(\tau|\pi) = \rho_0(s_0) \prod_{t=0}^{T-1}P(s_{t+1}|s_t,a_t)\pi_\theta(a_t|s_t)$$

### 3. Reward-Return and the Objective of RL:
+ reward is received after perform action $a_t$ at state $s_t$: $r_t=R(s_t,a_t)$.
+ return of a trajectory $\tau$ is the collected rewards $r_t$ weighted by a discount factor $\gamma$: $R(\tau) = \sum_{t=0}^T \gamma^t r_t$.
+ the goal of RL is to select a optimal policy $\pi^*$ which **maximizes expected return $J(\pi)$** when agent acts according to it:
    $$\pi^*= \argmax_\pi J(\pi) \quad \text{where} \quad J(\pi)= E_{\tau \sim \pi} [R(\tau)] = \int_\tau P(\tau|\pi) R(\tau) $$

### 4. Value function: 
If we start from a state $s$ or state-action pair $(s,a)$, and follows a particular policy $\pi$ forever after, we denote the **outcome return as the value** of $s$ or $(s,a)$:
+ Value function: $V^{\pi}(s) = E_{\tau \sim \pi} [R(\tau) | s_0=s]$ 
+ Action-Value function: $Q^{\pi}(s,a) = E_{\tau \sim \pi} [R(\tau) | s_0=s, a_0=a]$ 
+ Connection between $V^{\pi}(s)$ and $Q^{\pi}(s,a)$:
    $$ V^{\pi}(s) = E_{\tau \sim \pi} [R(\tau) | s_0=s] = \int_a \pi(a|s_0) E_{\tau \sim \pi} [R(\tau) | s_0=s, a_0=a] = E_{a \sim \pi}[Q^{\pi}(s,a)]$$
+ Advantage function: describe how much better to take a specific action $a$ in state $s$, over randomly selecting an action according to $\pi(\cdot|s)$: 
    $$ A^{\pi}(s,a)= Q^\pi(s,a) - V^{pi}(s) $$

+ If we follow the optimal policy at the state $s$, we call it as Optimal Value function $V^*(s)$.
+ If we follow the optimal policy after taking an action $a$ at the state $s$, we call it as Optimal Action-Value function $Q^*(s,a)$. 
+ Their connection is: $ V^*(s)= \max_a  Q^*(s,a)$.
+ The optimal action: $a^*(s) = \argmax_a Q^*(s,a)$.
  
## Kind of RL Algorithms:
Model-Free methods can divided into two-main approaches:
  + On-Policy: We learn the optimal policy $\pi_\theta(\cdot|s)$ directly, and always act according to the policy. The procedure generally includes :
    + Initialize random policy  $\pi_\theta(\cdot|s)$.
    + For each episode, do:
      + Follow the policy $\pi_\theta(a|s)$ to select action at each step %t$, and collect the rewards $r_t$ until the end of episode.
      + At the end of episode, compute the return $R(\tau)$, and the loss value $J(\pi)$. Update the policy by SGD.
            $$\theta \leftarrow \theta + \nappa J(\pi)$$
  + Off-Policy: We learn the Action Value function $Q_\theta(s,a)$, and indirectly select the optimal action that maximize $a^*(s) = \argmax_a Q^\theta(s,a)$. There is no direct policy in here, and the procedure generally includes:
    + Initialize random Q-value function:  $\Q^\theta(s,a)$.  
    + For each episode, do:
      + Select an action that maximize $Q(s,a)$, but with probability $\epsilon$, select the action randomly. Add the data $(s_t,a_t,r_t,s_{t+1},done)$ to the data buffer.
      + For every step, do sampling data from buffer, and update the $Q_\theta(s,a)$ network. 

The
    