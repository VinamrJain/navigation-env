---
title: "Discrete Grid Environment Formulation"
author: "Vinamr Jain"
date: "September 12, 2025"
toc: true
abstract: "A simplified discrete grid environment inspired by balloon navigation. The environment consists of a 3D grid with a spatially smooth stochastic wind field. Actors experience passive horizontal drift according to local wind distributions and controllable vertical movement. This formulation serves as a tractable abstraction for studying navigation and station-keeping policies under environmental uncertainty."
---
# Environment Formulation

The environment consists of two primary components: a **field** that governs environmental forces and an **actor** that moves according to both field dynamics and control actions.

## Field Definition

**Purpose**: The field represents environmental force distributions that passively advect actors in the horizontal plane. 

### Spatial Domain

**Grid Definition**: Define a discrete 3D grid:
$$\mathcal{G} = \{(i,j,k) \mid i \in \{1,\ldots,N_x\}, j \in \{1,\ldots,N_y\}, k \in \{1,\ldots,N_z\}\}$$
with unit spacing, representing the operational spatial domain.

\note{More generally, the defined grid need not be unit spaced.
Example: $\mathcal{G} = \{(x_i, y_j, z_k) \mid i \in \{1,...,N_x\}, j \in \{1,...,N_y\}, k \in \{1,...,N_p\}\}$ where $x_i \in \mathcal{X}, y_j \in \mathcal{Y}, z_k \in \mathcal{Z}$ over some domain}


**Displacement Space**: Define the set of possible horizontal displacements:
$$\mathcal{D} = \{(u,v) \mid u,v \in \mathbb{Z}, |u| \leq D_{\max}, |v| \leq D_{\max}\}$$
where $D_{\max} \in \mathbb{N}$ bounds the maximum displacement magnitude in each direction.

### Local Field Distribution

**Probability Mass Function**: For each grid point $(i,j,k) \in \mathcal{G}$, define a joint probability mass function over horizontal displacements:
$$p_{i,j,k}: \mathcal{D} \rightarrow [0,1]$$
$$p_{i,j,k}(u,v) = \mathbb{P}[U_{i,j,k} = u, V_{i,j,k} = v]$$

where $U_{i,j,k}, V_{i,j,k}$ are discrete random variables representing the horizontal displacement components experienced by an actor at grid point $(i,j,k)$.

### Field Properties and Constraints

The field must satisfy several mathematical constraints to ensure physical plausibility and computational tractability.

**Local Constraints** (for each $(i,j,k) \in \mathcal{G}$):

1. **Probability Mass Function Property**:
   $$\sum_{(u,v) \in \mathcal{D}} p_{i,j,k}(u,v) = 1, \quad p_{i,j,k}(u,v) \geq 0$$

2. **Boundary Conditions**: Displacements that would move an actor outside the grid are forbidden:
   $$p_{i,j,k}(u,v) = 0 \quad \text{if } \begin{cases}
   i + u \notin \{1,\ldots,N_x\} \\
   j + v \notin \{1,\ldots,N_y\}
   \end{cases}$$

3. **Variance Bounds**: Local displacement variability is bounded and specified:
   $$\text{Var}(U_{i,j,k}) \leq \sigma_u^2, \quad \text{Var}(V_{i,j,k}) \leq \sigma_v^2$$

4. **Covariance Structure**: The correlation between horizontal displacement components can be controlled:
   $$\text{Cov}(U_{i,j,k}, V_{i,j,k}) = \rho_{uv} \sigma_u \sigma_v$$
   where $\rho_{uv} \in [-1,1]$ is the correlation coefficient between $u$ and $v$ displacement components.

**Global Constraints**:

4. **Spatial Smoothness**: The displacement distributions must exhibit spatial coherence, meaning that neighboring grid points have similar statistical properties. This constraint ensures the field represents a physically plausible flow pattern rather than random noise.

   **Formal Requirement**: For neighboring grid points $(i,j,k), (i',j',k') \in \mathcal{G}$ with $\|(i,j,k) - (i',j',k')\|_1 = 1$, the probability mass functions $p_{i,j,k}$ and $p_{i',j',k'}$ must be "close" according to some distance metric $d(\cdot,\cdot)$:

   $$d(p_{i,j,k}, p_{i',j',k'}) \leq \epsilon$$

   where $\epsilon > 0$ is a smoothness parameter. The specific choice of distance metric $d(\cdot,\cdot)$ depends on the implementation method:
   - **Total Variation**: $d_{TV}(p,q) = \frac{1}{2}\sum_{(u,v)} |p(u,v) - q(u,v)|$
   - **Wasserstein**: $d_W(p,q)$ measuring transport cost between distributions
   - **KL-Divergence**: $D_{KL}(p||q)$ or Symmetric KL-Divergence (Jensen-shannon)
   - **Continuous Field Induced**: When PMFs are derived from an underlying smooth continuous field $\mathbf{c}(\mathbf{x})$, smoothness is inherited through the continuity properties of $\mathbf{c}(\mathbf{x})$

5. **Approximate Flow Conservation**: The expected flow should satisfy discrete incompressibility to maintain physical plausibility. Using central differences for interior points:
   $$\mathbb{E}\left[\frac{\partial U_{i,j,k}}{\partial x} + \frac{\partial V_{i,j,k}}{\partial y}\right] \approx 0$$
   $$\implies \left|\mathbb{E}[U_{i+1,j,k}] - \mathbb{E}[U_{i-1,j,k}] + \mathbb{E}[V_{i,j+1,k}] - \mathbb{E}[V_{i,j-1,k}]\right| \leq \delta$$

   where $\delta > 0$ is a tolerance parameter accounting for discretization errors and compressible effects.

**Parameterization**: The field is characterized by hyperparameters:
$$\Theta = \{N_x, N_y, N_z, D_{\max}, \sigma_u, \sigma_v, \rho_{uv},\epsilon, \delta, \text{seed}\}$$
where $\text{seed}$ enables reproducible stochastic generation of field realizations satisfying the above constraints.

<!-- Defined field should be ergodic? -->

## Actor Dynamics

**Control Actions**: The actor receives discrete control inputs:
$$\mathcal{A} = \{-1, 0, +1\}$$
corresponding to descend, maintain altitude, and ascend commands.

**Vertical Dynamics**: Define action-dependent vertical displacement PMF $q_a(z)$ for action $a \in \mathcal{A}$:
$$q_a(z) = \mathbb{P}[Z_a = z], \quad z \in \{-Z_{\max}, \ldots, Z_{\max}\}$$

where $Z_a$ is the vertical displacement random variable for action $a$.

\note{
\textbf{Example Action PMFs}: consider a simple parameterized vertical displacement distributions:

$$q_{+1}(z) = \begin{cases}
1-\epsilon & \text{if } z = +1 \\
\epsilon/2 & \text{if } z = 0 \\
\epsilon/2 & \text{if } z = +2 \\
0 & \text{otherwise}
\end{cases}$$

$$q_0(z) = \begin{cases}
1-\epsilon & \text{if } z = 0 \\
\epsilon/2 & \text{if } z = -1 \\
\epsilon/2 & \text{if } z = +1 \\
0 & \text{otherwise}
\end{cases}$$

$$q_{-1}(z) = \begin{cases}
1-\epsilon & \text{if } z = -1 \\
\epsilon/2 & \text{if } z = 0 \\
\epsilon/2 & \text{if } z = -2 \\
0 & \text{otherwise}
\end{cases}$$

where $\epsilon \in [0,1]$ is a noise parameter controlling the reliability of vertical control.}

**State Transition**: Given actor position $(i,j,k)$ and action $a$, the new position $(i',j',k')$ is determined by:
1. **Horizontal Displacement**: $(u,v) \sim p_{i,j,k}(\cdot,\cdot)$
2. **Vertical Displacement**: $z \sim q_a(\cdot)$
3. **Position Update**:
   $$i' = i + u, \quad j' = j + v, \quad k' = k + z$$

**Boundary Handling**: Ensure transitions remain within grid bounds:
$$(i',j',k') \in \mathcal{G}$$


\note{
\textbf{Boundary Enforcement example:} If a transition would violate grid bounds, enforce hard boundary conditions:
$$i' = \max(1, \min(i + u, N_x))$$
$$j' = \max(1, \min(j + v, N_y))$$
$$k' = \max(1, \min(k + z, N_z))$$
}


<!--
wind field Abstract class definition

reset
get_wind

wind field instantiations

wind vector abstract class

u, v

actor abstract class

actor_state (not needed)
- last action
- last reward
- last wind observation at point

actor_step{action, wind_vector}
 specify up or down pmf

arena interface if needed
gym environment

**Key considerations:**
1. For the horizontal displacement, in a continuous setting it can be interpolated using the new point and the old point (eg. newton interpolation)

END OF ORIGINAL ROUGH NOTES
-->