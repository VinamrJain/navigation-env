---
title: "Grid Environment POMDP Formulation"
author: "Vinamr Jain"
date: "September 15, 2025"
toc: true
abstract: "POMDP formulation for the discrete grid environment with stochastic partially observable static fields."
---

<!--

# Thoughts and comments
\note{Sidenote:  and compute optimal policy if the agent knows the underlying state}
\note{Corollary extension: suppose agent knows an approximation (not the exact) of the underlying state, what's the best policy?}
\note{What would state estimation look like in this scenario?}
-->

# POMDP Formulation

## State Space ($\mathcal{S}$)

The system state $s_t \in \mathcal{S}$ at time $t$ is a composite tuple containing the actor's position and the complete field configuration:

$$s_t = (\mathbf{r}_t, \boldsymbol{\Phi})$$

where:

### Actor State ($\mathbf{r}_t$)

The actor's spatial configuration is defined by its grid position:
$$\mathbf{r}_t = (i_t, j_t, k_t) \in \mathcal{G}$$

where $(i_t, j_t, k_t)$ represents the discrete coordinates within the 3D grid $\mathcal{G} = \{(i,j,k) \mid i \in \{1,\ldots,N_x\}, j \in \{1,\ldots,N_y\}, k \in \{1,\ldots,N_z\}\}$.

### Field State ($\boldsymbol{\Phi}$)

**Gaussian Parameterization**: To achieve computational tractability, we approximate the local displacement distributions as discretized bivariate Gaussian distributions. The field state is parameterized by means and covariances at each grid point:

$$\boldsymbol{\Phi} = \{(\boldsymbol{\mu}_{i,j,k}, \boldsymbol{\Sigma}_{i,j,k}) \mid (i,j,k) \in \mathcal{G}\}$$

where:
- $\boldsymbol{\mu}_{i,j,k} = [\mu_{u,i,j,k}, \mu_{v,i,j,k}]^T \in \mathbb{R}^2$ is the mean displacement vector
- $\boldsymbol{\Sigma}_{i,j,k} = \begin{pmatrix} \sigma_{u,i,j,k}^2 & \rho_{i,j,k}\sigma_{u,i,j,k}\sigma_{v,i,j,k} \\ \rho_{i,j,k}\sigma_{u,i,j,k}\sigma_{v,i,j,k} & \sigma_{v,i,j,k}^2 \end{pmatrix}$ is the covariance matrix

**Discretized Gaussian PMF**: The local displacement PMF is constructed as:
$$p_{i,j,k}(u,v) = \frac{\mathcal{N}((u,v) \mid \boldsymbol{\mu}_{i,j,k}, \boldsymbol{\Sigma}_{i,j,k}) \cdot \mathbb{I}_{i,j,k}(u,v)}{Z_{i,j,k}}$$

where $\mathbb{I}_{i,j,k}(u,v) = 1$ if $(i+u, j+v) \in [1,N_x] \times [1,N_y]$ (boundary constraint), and $Z_{i,j,k}$ is the normalization constant.

**State Space Cardinality**: Under Gaussian parameterization:
$$|\mathcal{S}| = N_x N_y N_z \times \prod_{(i,j,k) \in \mathcal{G}} |\boldsymbol{\Theta}_{i,j,k}|$$

where $\boldsymbol{\Theta}_{i,j,k} = (\boldsymbol{\mu}_{i,j,k}, \boldsymbol{\Sigma}_{i,j,k})$ represents the 5-dimensional parameter space (2 means + 3 unique covariance elements) for each grid point. For continuous parameters, this requires discretization of the parameter space.

**Tractable Alternative**: For computational tractability, assume a **common covariance structure** across grid points:
$$\boldsymbol{\Sigma}_{i,j,k} = \boldsymbol{\Sigma}_{\text{global}} \quad \forall (i,j,k) \in \mathcal{G}$$

This reduces the field state to mean displacements only:
$$\boldsymbol{\Phi}_{\text{reduced}} = \{\boldsymbol{\mu}_{i,j,k} \mid (i,j,k) \in \mathcal{G}\}$$

with state space cardinality $|\mathcal{S}| = N_x N_y N_z \times \prod_{(i,j,k) \in \mathcal{G}} |\boldsymbol{\mu}_{i,j,k}|$ where each mean vector requires discretization over a 2D grid.

<!-- **Explanation of State Space Cardinality**:

The state space cardinality formula $|\mathcal{S}| = N_x N_y N_z \times \prod_{(i,j,k) \in \mathcal{G}} |\text{support}(p_{i,j,k})|$ decomposes as follows:

1. **Actor Position Component**: $N_x N_y N_z$ represents the number of possible actor positions, as the actor can be at any of the $N_x \times N_y \times N_z$ grid points.

2. **Field Configuration Component**: $\prod_{(i,j,k) \in \mathcal{G}} |\text{support}(p_{i,j,k})|$ counts distinct field configurations. Each grid point $(i,j,k)$ has a PMF $p_{i,j,k}$ that can assign positive probability to any subset of displacement vectors in $\mathcal{D}$. The number of possible supports for each PMF is $|\text{support}(p_{i,j,k})| \leq 2^{|\mathcal{D}|}$, and since PMFs at different grid points can vary independently, we take the product over all grid points.

**Practical Bounds**: 
- **Lower bound**: $|\mathcal{S}| \geq N_x N_y N_z$ (when all PMFs are deterministic)
- **Upper bound**: $|\mathcal{S}| \leq N_x N_y N_z \times (2^{|\mathcal{D}|})^{N_x N_y N_z}$ (when any subset of $\mathcal{D}$ can have positive probability) -->


## Action Space ($\mathcal{A}$)

The action space consists of discrete vertical control commands:
$$\mathcal{A} = \{a_{down}, a_{stay}, a_{up}\} = \{-1, 0, +1\}$$

**Action Semantics**:
- $a = -1$: **Descend** - Initiate downward movement
- $a = 0$: **Stay** - Maintain current altitude level
- $a = +1$: **Ascend** - Initiate upward movement

## Transition Dynamics ($T$)

The transition function $T(s_{t+1} \mid s_t, a_t)$ defines the probability distribution over next states given current state and action. Due to the static nature of the field and the decoupled dynamics, this factors as:

$$P(s_{t+1} \mid s_t, a_t) = P(\boldsymbol{\Phi}_{t+1} \mid \boldsymbol{\Phi}_t) \cdot P(\mathbf{r}_{t+1} \mid \mathbf{r}_t, \boldsymbol{\Phi}_t, a_t)$$

### Field Dynamics

**Static Field Assumption**: The environmental field remains constant throughout the episode:
$$P(\boldsymbol{\Phi}_{t+1} \mid \boldsymbol{\Phi}_t) = \delta(\boldsymbol{\Phi}_{t+1} - \boldsymbol{\Phi}_t)$$

where $\delta(\cdot)$ is the Dirac delta function, ensuring $\boldsymbol{\Phi}_{t+1} = \boldsymbol{\Phi}_t$ with probability 1.

### Actor Dynamics

The actor's position evolves through the coupled horizontal-vertical displacement process:

**Horizontal Displacement**: Given current position $(i_t, j_t, k_t)$, sample horizontal displacement from the local field PMF:
$$(u_t, v_t) \sim p_{i_t,j_t,k_t}(\cdot, \cdot)$$

**Vertical Displacement**: Sample vertical displacement from action-dependent PMF:
$$z_t \sim q_{a_t}(\cdot)$$

where $q_{a_t}: \{-Z_{\max}, \ldots, Z_{\max}\} \rightarrow [0,1]$ is the vertical displacement PMF for action $a_t$.

**Position Update**: The next position is computed with boundary enforcement:
$$i_{t+1} = \text{clip}(i_t + u_t, 1, N_x)$$
$$j_{t+1} = \text{clip}(j_t + v_t, 1, N_y)$$
$$k_{t+1} = \text{clip}(k_t + z_t, 1, N_z)$$

where $\text{clip}(x, a, b) = \max(a, \min(x, b))$ enforces grid boundaries.

**Transition Probability**: The complete actor transition probability is:
$$P(\mathbf{r}_{t+1} \mid \mathbf{r}_t, \boldsymbol{\Phi}_t, a_t) = p_{i_t,j_t,k_t}(u_t, v_t) \cdot q_{a_t}(z_t) \cdot \mathbb{I}_{\text{valid}}(\mathbf{r}_{t+1})$$

where $u_t = i_{t+1} - i_t$, $v_t = j_{t+1} - j_t$, $z_t = k_{t+1} - k_t$ (before clipping), and $\mathbb{I}_{\text{valid}}(\mathbf{r}_{t+1})$ is an indicator function ensuring the final position lies within $\mathcal{G}$.

## Observation Space ($\mathcal{O}$)

The observation $o_t \in \mathcal{O}$ represents the agent's sensory information at time $t$:

$$o_t = (\mathbf{r}_t^{\text{obs}}, \mathbf{w}_t^{\text{local}})$$

where:

### Position Observation ($\mathbf{r}_t^{\text{obs}}$)

**Perfect Localization**: The agent has perfect knowledge of its current position:
$$\mathbf{r}_t^{\text{obs}} = \mathbf{r}_t = (i_t, j_t, k_t)$$

### Local Field Observation ($\mathbf{w}_t^{\text{local}}$)

**Point-wise Field Sampling**: The agent observes the realized displacement from the local field distribution:

$$\mathbf{w}_t^{\text{local}} = (u_t^{\text{realized}}, v_t^{\text{realized}})$$

where $(u_t^{\text{realized}}, v_t^{\text{realized}})$ is the actual displacement experienced during the transition, sampled from the discretized Gaussian PMF:
$$(u_t^{\text{realized}}, v_t^{\text{realized}}) \sim p_{i_t,j_t,k_t}(\cdot, \cdot)$$

This represents the "wind measurement" - the observed effect of the local field on the actor's movement.

<!-- **Alternative Observation Models**:

1. **Neighborhood Sampling**: Observe displacements from neighboring grid points:
   $$\mathbf{w}_t^{\text{local}} = \{(u^{\text{obs}}, v^{\text{obs}}) \sim p_{i',j',k'} : \|(i',j',k') - (i_t,j_t,k_t)\|_1 \leq R\}$$

2. **Statistical Moments**: Observe empirical statistics rather than raw samples:
   $$\mathbf{w}_t^{\text{local}} = (\hat{\mu}_u, \hat{\mu}_v, \hat{\sigma}_u^2, \hat{\sigma}_v^2, \hat{\rho}_{uv})$$ -->

## Observation Model ($O$)

The observation model $O(o_t \mid s_t)$ specifies the probability of receiving observation $o_t$ given true state $s_t$. Due to independence assumptions:

$$P(o_t \mid s_t) = P(\mathbf{r}_t^{\text{obs}} \mid \mathbf{r}_t) \cdot P(\mathbf{w}_t^{\text{local}} \mid \mathbf{r}_t, \boldsymbol{\Phi})$$

### Position Observation Model

Under perfect localization:
$$P(\mathbf{r}_t^{\text{obs}} \mid \mathbf{r}_t) = \delta(\mathbf{r}_t^{\text{obs}} - \mathbf{r}_t)$$

### Field Observation Model

For realized displacement observation at position $(i_t, j_t, k_t)$:
$$P(\mathbf{w}_t^{\text{local}} = (u,v) \mid \mathbf{r}_t, \boldsymbol{\Phi}) = p_{i_t,j_t,k_t}(u,v) = \frac{\mathcal{N}((u,v) \mid \boldsymbol{\mu}_{i_t,j_t,k_t}, \boldsymbol{\Sigma}_{i_t,j_t,k_t}) \cdot \mathbb{I}_{i_t,j_t,k_t}(u,v)}{Z_{i_t,j_t,k_t}}$$

## Belief State ($\mathcal{B}$)

The belief state $b_t$ is a probability distribution over the state space, conditioned on the history of actions and observations $h_t = (o_1, a_1, o_2, a_2, \ldots, a_{t-1}, o_t)$:

$$b_t(s) = P(s_t = s \mid h_t)$$

### Factored Belief Representation

Due to perfect position observation and static field assumption:
$$b_t(s) = b_t(\mathbf{r}, \boldsymbol{\Phi}) = \delta(\mathbf{r} - \mathbf{r}_t) \cdot b_t(\boldsymbol{\Phi})$$

The belief update reduces to maintaining uncertainty only over the field configuration $\boldsymbol{\Phi}$.

### Gaussian Field Belief Model

**Conjugate Prior Structure**: For the Gaussian-parameterized field, we employ a **Normal-Inverse-Wishart** conjugate prior structure for each grid point. Under the common covariance assumption $\boldsymbol{\Sigma}_{i,j,k} = \boldsymbol{\Sigma}_{\text{global}}$, this simplifies to independent Normal priors on mean displacements:

$$\boldsymbol{\mu}_{i,j,k} \mid \text{observations} \sim \mathcal{N}(\hat{\boldsymbol{\mu}}_{i,j,k}, \boldsymbol{\Sigma}_{posterior,i,j,k})$$

**Prior Specification**: Initialize with uninformative priors:
$$\boldsymbol{\mu}_{i,j,k} \sim \mathcal{N}(\mathbf{0}, \sigma_{prior}^2 \mathbf{I})$$

where $\sigma_{prior}^2$ reflects prior uncertainty about displacement magnitudes.

**Belief Update**: Upon observing displacement $(u^{obs}, v^{obs})$ at position $(i,j,k)$:

1. **Likelihood**: The observation likelihood under current belief parameters:
   $$\mathcal{L}(\boldsymbol{\mu}_{i,j,k}) \propto \mathcal{N}((u^{obs}, v^{obs}) \mid \boldsymbol{\mu}_{i,j,k}, \boldsymbol{\Sigma}_{\text{global}})$$

2. **Posterior Update**: Bayesian update with Normal-Normal conjugacy:
   $$\hat{\boldsymbol{\mu}}_{i,j,k}^{new} = \frac{\boldsymbol{\Sigma}_{\text{global}}^{-1} (u^{obs}, v^{obs})^T + \boldsymbol{\Sigma}_{posterior,i,j,k}^{-1} \hat{\boldsymbol{\mu}}_{i,j,k}^{old}}{\boldsymbol{\Sigma}_{\text{global}}^{-1} + \boldsymbol{\Sigma}_{posterior,i,j,k}^{-1}}$$

   $$\boldsymbol{\Sigma}_{posterior,i,j,k}^{new} = \left(\boldsymbol{\Sigma}_{\text{global}}^{-1} + \boldsymbol{\Sigma}_{posterior,i,j,k}^{-1}\right)^{-1}$$

**Computational Advantages**:
- **Analytical Updates**: Closed-form belief updates without sampling
- **Factorization**: Independent beliefs per grid point (under static field assumption)
- **Uncertainty Quantification**: Explicit posterior covariances for exploration-exploitation trade-offs

### Alternative: Particle Filter Belief

For non-Gaussian or spatially correlated field beliefs:

**Particle Representation**: Approximate the field belief through weighted samples:
$$b_t(\boldsymbol{\Phi}) \approx \sum_{i=1}^{N_p} w_t^{(i)} \delta(\boldsymbol{\Phi} - \boldsymbol{\Phi}^{(i)})$$

where $\{(\boldsymbol{\Phi}^{(i)}, w_t^{(i)})\}_{i=1}^{N_p}$ are field configuration particles with weights.

**Particle Update**: Standard particle filter operations:
1. **Prediction**: $\boldsymbol{\Phi}_{t+1}^{(i)} = \boldsymbol{\Phi}_t^{(i)}$ (static field)
2. **Weight Update**: $w_{t+1}^{(i)} \propto w_t^{(i)} \cdot P(\mathbf{w}_t^{\text{local}} \mid \mathbf{r}_t, \boldsymbol{\Phi}_t^{(i)})$
3. **Resampling**: When effective sample size drops below threshold

<!-- ### Field Belief Parameterization

**Dirichlet-Categorical Model**: For each grid point $(i,j,k)$, maintain a Dirichlet posterior over the PMF parameters:

$$p_{i,j,k} \mid \text{observations} \sim \text{Dir}(\boldsymbol{\alpha}_{i,j,k})$$

where $\boldsymbol{\alpha}_{i,j,k} \in \mathbb{R}_+^{|\mathcal{D}|}$ are the Dirichlet concentration parameters.

**Belief Update**: Upon observing $(u,v)$ at position $(i,j,k)$:
$$\boldsymbol{\alpha}_{i,j,k} \leftarrow \boldsymbol{\alpha}_{i,j,k} + \mathbf{e}_{(u,v)}$$

where $\mathbf{e}_{(u,v)}$ is the unit vector corresponding to displacement $(u,v)$. -->

## Reward Function ($R$)

The reward function $R(s_t, a_t)$ provides the learning signal aligned with the navigation objective:

$$R: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$$

<!-- ### Station-Keeping Objective

For station-keeping around a target position $\mathbf{r}^* = (i^*, j^*, k^*)$:

$$R_{\text{station}}(s_t, a_t) = -\|\mathbf{r}_t - \mathbf{r}^*\|_2^2 - \lambda_a \mathbb{I}[a_t \neq 0] - \lambda_b \mathbb{I}[\mathbf{r}_t \notin \mathcal{R}_{\text{safe}}]$$

where:
- $\lambda_a > 0$ penalizes unnecessary control actions
- $\lambda_b \gg 1$ heavily penalizes boundary violations
- $\mathcal{R}_{\text{safe}} \subset \mathcal{G}$ is the safe operating region

### Navigation Objective

For point-to-point navigation from start $\mathbf{r}_0$ to goal $\mathbf{r}^{\text{goal}}$:

$$R_{\text{nav}}(s_t, a_t) = \begin{cases}
+R_{\text{goal}} & \text{if } \mathbf{r}_t = \mathbf{r}^{\text{goal}} \\
-\|\mathbf{r}_t - \mathbf{r}^{\text{goal}}\|_1 - \lambda_t & \text{otherwise}
\end{cases}$$

where $R_{\text{goal}} > 0$ is the goal reward and $\lambda_t > 0$ provides time penalty to encourage efficiency. -->

## Optimization Objective

The agent seeks an optimal policy $\pi^*: \mathcal{B} \rightarrow \mathcal{A}$ that maximizes expected cumulative discounted reward:

$$\pi^* = \arg\max_{\pi} J(\pi)$$

where:
$$J(\pi) = \mathbb{E}_{\pi}\left[\sum_{t=0}^{T-1} \gamma^t R(s_t, a_t) \mid b_0\right]$$

with discount factor $\gamma \in (0,1)$ and finite horizon $T$ or infinite horizon ($T \to \infty$).

## Computational Tractability Analysis

### State Space Complexity Under Gaussian Parameterization

**Reduced Complexity**: The Gaussian parameterization significantly reduces state space complexity compared to arbitrary PMF specification:

- **Original PMF approach**: $|\mathcal{S}| = N_x N_y N_z \times \prod_{(i,j,k)} 2^{|\mathcal{D}|}$ (exponential in displacement space size)
- **Gaussian approach**: $|\mathcal{S}| = N_x N_y N_z \times \prod_{(i,j,k)} |\boldsymbol{\mu}_{i,j,k}|$ (polynomial in discretization resolution)

**Further Reduction with Common Covariance**: Under $\boldsymbol{\Sigma}_{i,j,k} = \boldsymbol{\Sigma}_{\text{global}}$:
$$|\mathcal{S}| = N_x N_y N_z \times (N_{\mu})^{2 \cdot N_x N_y N_z}$$

where $N_{\mu}$ is the discretization resolution for each mean component.

### Alternative Wind Field Modeling Approaches

**Approach 1: Low-Rank Field Approximation**

Represent the field means as a linear combination of spatial basis functions:
$$\boldsymbol{\mu}_{i,j,k} = \sum_{\ell=1}^{L} \alpha_\ell \boldsymbol{\phi}_\ell(i,j,k)$$

where $\{\boldsymbol{\phi}_\ell\}$ are spatial basis functions (e.g., DCT, wavelets, or spatial eigenmodes) and $\{\alpha_\ell\}$ are coefficients.

**State Space Reduction**: The field state reduces to:
$$\boldsymbol{\Phi}_{\text{low-rank}} = \{\alpha_1, \alpha_2, \ldots, \alpha_L\}$$

with state space cardinality $|\mathcal{S}| = N_x N_y N_z \times (N_{\alpha})^L$ where $L \ll 2 N_x N_y N_z$.

**Approach 2: Spatial Correlation Model**

Model the field as a spatially correlated Gaussian process:
$$\boldsymbol{\mu} = [\boldsymbol{\mu}_{1,1,1}^T, \ldots, \boldsymbol{\mu}_{N_x,N_y,N_z}^T]^T \sim \mathcal{N}(\mathbf{0}, \mathbf{K})$$

where $\mathbf{K}$ encodes spatial correlations with kernel function $k(\mathbf{r}, \mathbf{r}')$.

**Belief Update**: Use Gaussian process posterior updates:
$$\boldsymbol{\mu} \mid \text{observations} \sim \mathcal{N}(\boldsymbol{\mu}_{\text{post}}, \mathbf{K}_{\text{post}})$$

**Computational Benefit**: Leverages spatial correlations to reduce effective dimensionality and enable efficient inference.

**Approach 3: Hierarchical Field Model**

Use a **coarse-to-fine hierarchical representation**:

1. **Coarse Level**: Maintain beliefs over field parameters on a coarse grid $(N_x^c, N_y^c, N_z^c)$ where $N_i^c \ll N_i$
2. **Fine Level**: Interpolate/extrapolate coarse parameters to fine grid when needed for local decisions

**State Space**: $|\mathcal{S}| = N_x N_y N_z \times (N_{\mu})^{2 \cdot N_x^c N_y^c N_z^c}$

## Recommended Implementation Strategy

**Phase 1: Proof of Concept**
- Gaussian parameterization with common covariance $\boldsymbol{\Sigma}_{\text{global}}$
- Small grid ($5 \times 5 \times 3$)
- Conjugate Normal priors for analytical belief updates

**Phase 2: Scalability**
- Low-rank field approximation with $L = 10-20$ basis functions
- Medium grid ($10 \times 10 \times 5$)
- Compare analytical vs. particle filter belief updates

**Phase 3: Realism**
- Spatial correlation model or hierarchical representation
- Large grid ($20 \times 20 \times 10$)
- Integration with balloon dynamics from your BLE formulation

<!-- ## Computational Considerations

### State Space Complexity

**Curse of Dimensionality**: The field state space $\boldsymbol{\Phi}$ is exponential in grid size, making exact solution methods intractable for realistic problem instances.

**Belief Space Dimensionality**: Even with factored representation, the Dirichlet belief parameterization requires $O(N_x N_y N_z |\mathcal{D}|)$ parameters.

### Approximate Solution Methods

**Particle Filter POMDP**: Represent beliefs through weighted particle sets:
$$b_t \approx \{(\boldsymbol{\Phi}^{(i)}, w^{(i)}) : i = 1, \ldots, K\}$$

**Monte Carlo Tree Search**: Use POMCP or similar algorithms with:
- Particle-based belief representation
- Progressive widening for action selection
- UCB-based exploration strategy

**Policy Gradient Methods**: Learn parameterized policies $\pi_\theta(a \mid b)$ through:
- Actor-critic architectures with recurrent belief encoding
- Experience replay with belief state approximation
- Regularization to prevent overfitting to specific field configurations

--- -->

**Key Assumptions Summary**:

1. **Static Field**: Environmental field configuration remains constant throughout episodes
2. **Perfect Localization**: Agent position is always perfectly observable
3. **Markovian Transitions**: Current position and field state sufficient for predicting next position
4. **Finite Displacement Bounds**: All displacements bounded by $D_{\max}$ and $Z_{\max}$
5. **Boundary Reflection**: Hard boundaries with clipping/reflection at domain edges

<!-- 4. **Independent Field Points**: PMFs at different grid locations evolve independently (under Dirichlet model) -->