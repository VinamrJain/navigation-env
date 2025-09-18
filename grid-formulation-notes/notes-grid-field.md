---
title: "Grid Field Formulation"
author: "Vinamr Jain"
date: "September 15, 2025"
toc: true
---
# Field Implementation Strategy

## Design Philosophy

The implementation follows a two-stage paradigm: first generate a spatially smooth continuous field that satisfies global constraints, then discretize it into probability mass functions while enforcing local constraints.

## Stage 1: Continuous Field Generation

**Continuous Field Formulation**: The continuous displacement field $\mathbf{c}(\mathbf{r}) = (c_u(\mathbf{r}), c_v(\mathbf{r}))$ is defined over the spatial domain, where $\mathbf{r} = (x,y,z)$ represents continuous spatial coordinates. At discrete grid points, we evaluate:
$$\mathbf{c}_{i,j,k} = \mathbf{c}(\mathbf{r}_{i,j,k}) = (c_u(\mathbf{r}_{i,j,k}), c_v(\mathbf{r}_{i,j,k}))$$

where $\mathbf{r}_{i,j,k}$ are the physical coordinates corresponding to grid indices $(i,j,k)$. Spatial smoothness is ensured by construction through the continuous field's inherent smoothness properties and spatial correlation structure. The constructed field must satisfy approximate flow conservation. 

<!-- **Global Constraint Satisfaction**:
- **Spatial Smoothness**: Ensured by construction through the continuous field's inherent smoothness properties and spatial correlation structure
- **Approximate Flow Conservation**: 
  - **Method I (Streamfunction)**: Satisfies $\nabla \cdot \mathbf{c} = 0$ exactly by construction via $\mathbf{c} = \nabla \times \psi$
  - **Method II (Constrained Gaussian)**: Enforces discrete divergence-free condition $\mathbf{A}\boldsymbol{\mu} = \mathbf{0}$ through linear constraints on the displacement field means -->


## Stage 2: Construct Local PMFs from the Continuous Field
**Input**: A continuous vector field $\mathbf{c}_{i,j,k} = (c_u, c_v)_{i,j,k}$

**Output**: A valid PMF $p_{i,j,k}(u,v)$ at each grid point.

**Define Local Weights**: For each grid point $(i,j,k)$, use the vector $\mathbf{c}_{i,j,k}$ as the mean of a local bivariate normal PDF $\mathcal{N}(\cdot \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}_{\text{local}})$. Calculate weights for all $(u,v) \in \mathcal{D}$:

$$w_{ijk}(u,v) = \mathcal{N}((u,v) \mid \boldsymbol{\mu} = \mathbf{c}_{ijk}, \boldsymbol{\Sigma}_{\text{local}})$$

**Enforce Boundary Conditions**: Define an indicator function $I_{ijk}(u,v)$ that is 1 if the displacement $(u,v)$ from $(i,j)$ is within the grid boundaries, and 0 otherwise. Apply it:

$$w'_{ijk}(u,v) = w_{ijk}(u,v) \cdot I_{ijk}(u,v)$$

where, $\mathbb{I}_{i,j,k}(u,v) = 1$ if $(i+u, j+v) \in [1,N_x] \times [1,N_y]$, and $0$ otherwise.

**Normalize**: Calculate the normalization constant $Z_{ijk} = \sum_{(u,v) \in \mathcal{D}} w'_{ijk}(u,v)$. The final PMF is:

$$p_{i,j,k}(u,v) = \frac{w'_{i,j,k}(u,v)}{Z_{i,j,k}}$$

**Local Constraints**:
- **PMF Properties**:  The weights $w'_{i,j,k}(u,v) \geq 0$ are non-negative by construction  and normalization ensures $\sum_{(u,v) \in \mathcal{D}} p_{i,j,k}(u,v) = 1$.

- **Boundary Conditions**: Enforced exactly by the indicator function $\mathbb{I}_{i,j,k}(u,v)$. When $(i+u, j+v) \notin [1,N_x] \times [1,N_y]$, we have $\mathbb{I}_{i,j,k}(u,v) = 0$, which zeroes out $w'_{i,j,k}(u,v)$ and consequently $p_{i,j,k}(u,v) = 0$.

- **Variance Bounds**: Controlled by choice of $\boldsymbol{\Sigma}_{\text{local}}$. Define the local covariance matrix as:
  $$\boldsymbol{\Sigma}_{\text{local}} = \begin{pmatrix} \sigma_u^2 & \rho_{uv}\sigma_u\sigma_v \\ \rho_{uv}\sigma_u\sigma_v & \sigma_v^2 \end{pmatrix}$$
  
  The resulting PMF $p_{i,j,k}(u,v)$ is a discretized and renormalized version of $\mathcal{N}(\mathbf{c}_{i,j,k}, \boldsymbol{\Sigma}_{\text{local}})$. While boundary truncation and renormalization modify the exact variance, the choice of $\boldsymbol{\Sigma}_{\text{local}}$ directly controls the concentration of probability mass around the mean $\mathbf{c}_{i,j,k}$, allowing approximate satisfaction of the variance bounds through appropriate parameter selection.

- **Covariance Structure**: The cross-correlation $\rho_{uv}$ in $\boldsymbol{\Sigma}_{\text{local}}$ directly controls the correlation between $U_{i,j,k}$ and $V_{i,j,k}$ components in the resulting PMF, subject to boundary effects and renormalization.

<!-- - Discretized multivariate gaussian (less parameters to specify (u,\sigma instead of 2T+1 parameters)) -->

# Continuous Field Generation

## Method I: Streamfunction-Based Incompressible Fields

### General Framework for Incompressible Field Generation

**Theoretical Foundation**: Any divergence-free vector field in two dimensions can be expressed in terms of a scalar streamfunction. This provides a systematic approach for generating incompressible displacement fields that maintains exact flow conservation while preserving spatial correlation properties.

 Given any continuous scalar field $\Psi(\mathbf{r})$ defined over the spatial domain $\mathbf{r} = (x, y, z)$, we can construct an exactly incompressible displacement field via:
$$\mathbf{c}(\mathbf{r}) = \nabla \times (\Psi(\mathbf{r}) \hat{\mathbf{z}}) = \left(-\frac{\partial \Psi}{\partial y}, \frac{\partial \Psi}{\partial x}\right)$$

This construction automatically satisfies $\nabla \cdot \mathbf{c} = 0$ since:
$$\nabla \cdot \mathbf{c} = -\frac{\partial^2 \Psi}{\partial x \partial y} + \frac{\partial^2 \Psi}{\partial y \partial x} = 0$$

**Implementation Pipeline**:
1. **Continuous Field Specification**: Define a continuous scalar field $\Psi(\mathbf{r})$ with desired spatial correlation and smoothness properties
2. **Extended Grid Sampling**: Evaluate $\Psi$ on an extended  $N =(N_x+1) \times (N_y+1) \times N_z$ grid to support boundary derivative calculations
3. **Discrete Differentiation**: Apply central difference operators to compute displacement components on the interior $N_x \times N_y \times N_z$ grid
4. **Linear Transformation**: Express the velocity derivation as matrix operations

**Discrete Grid Evaluation**: For grid coordinates $\mathbf{r}_{i,j,k} = (i \Delta x, j \Delta y, k \Delta z)$, sample the streamfunction on the extended grid:
$$\boldsymbol{\Psi} = [\Psi_{0,0,1}, \Psi_{0,1,1}, \ldots, \Psi_{N_x,N_y,N_z}]^T$$

**Central Difference Implementation**: Compute displacement field components for interior points $(i,j,k) \in \{1, \ldots, N_x\} \times \{1, \ldots, N_y\} \times \{1, \ldots, N_z\}$:

**Horizontal Component ($u$-direction)**:
$$c_{u,i,j,k} = -\frac{\partial \Psi}{\partial y}\bigg|_{i,j,k} \approx -\frac{\Psi_{i,j+1,k} - \Psi_{i,j-1,k}}{2\Delta y}$$

**Vertical Component ($v$-direction)**:
$$c_{v,i,j,k} = \frac{\partial \Psi}{\partial x}\bigg|_{i,j,k} \approx \frac{\Psi_{i+1,j,k} - \Psi_{i-1,j,k}}{2\Delta x}$$

**Matrix Formulation**: Express as linear transformations:
$$\mathbf{c}_u = -\frac{1}{2\Delta y}\mathbf{D}_y \boldsymbol{\Psi}, \quad \mathbf{c}_v = \frac{1}{2\Delta x}\mathbf{D}_x \boldsymbol{\Psi}$$

where $\mathbf{D}_x$ and $\mathbf{D}_y$ are $N \times (N+1)$ sparse matrices encoding central difference operators, mapping from the extended streamfunction grid to the interior displacement field.

## Method Ia: Gaussian Process Streamfunction

**Continuous Streamfunction Model**: Define the streamfunction as a Gaussian random field:
$$\Psi(\mathbf{r}) \sim \mathcal{GP}(m(\mathbf{r}), K(\mathbf{r}, \mathbf{r}'))$$

**Statistical Specification**:
- **Mean Function**: $m(\mathbf{r})$
- **Covariance Kernel**: $K(\mathbf{r}, \mathbf{r}') = \sigma_\Psi^2 \exp\left(-\frac{\|\mathbf{r} - \mathbf{r}'\|^2}{2\ell^2}\right)$
- **Hyperparameters**: $\sigma_\Psi^2$ controls field variance, $\ell$ controls spatial correlation length

**Discrete Sampling**: Generate streamfunction values on the extended grid:
$$\boldsymbol{\Psi} \sim \mathcal{N}(\mathbf{0}, \mathbf{K})$$

where $[\mathbf{K}]_{(i,j,k),(i',j',k')} = K(\mathbf{r}_{i,j,k}, \mathbf{r}_{i',j',k'})$ is the covariance matrix for $(N_x+1) \times (N_y+1) \times N_z$ grid points.

**Resulting Field Properties**:
- **Exact Incompressibility**: $\nabla \cdot \mathbf{c} = 0$ by construction
- **Spatial Smoothness**: Inherited from Gaussian process smoothness (infinitely differentiable for squared exponential kernel)
- **Statistical Structure**: Velocity components $\mathbf{c}_u$ and $\mathbf{c}_v$ are jointly Gaussian with covariance determined by streamfunction kernel and difference operators

## Method Ib: Simplex Noise Streamfunction

**Multi-Octave Streamfunction Construction**: Define the streamfunction using simplex noise with fractal structure:
$$\Psi(\mathbf{r}) = \sum_{m=0}^{M-1} A_m \eta_\Psi(2^m \mathbf{r} / L)$$

where $\eta_\Psi(\mathbf{r})$ is a base simplex noise function with the multi-octave parameters:

**Octave Parameters**:
- $M$: Number of octaves controlling field complexity
- $A_m = A_0 \beta^m$: Amplitude decay with persistence parameter $\beta \in (0,1)$
- $L$: Base length scale controlling spatial correlation
- $2^m$: Frequency doubling between octaves

**Amplitude Normalization**: For consistent field statistics:
$$A_0 = \sigma_\Psi \left(\sum_{m=0}^{M-1} \beta^{2m}\right)^{-1/2}$$

where $\sigma_\Psi$ is the desired streamfunction standard deviation.

**Grid Evaluation**: Sample the streamfunction at extended grid points:
$$\Psi_{i,j,k} = \sum_{m=0}^{M-1} A_m \eta_\Psi(2^m (i,j,k) / L)$$

**Resulting Field Properties**:
- **Exact Incompressibility**: $\nabla \cdot \mathbf{c} = 0$ by streamfunction construction
- **Fractal Structure**: Multi-octave construction provides natural turbulence-like patterns
- **Computational Efficiency**: Direct evaluation without matrix operations
<!-- - **Controlled Smoothness**: $C^{k-1}$ smoothness determined by simplex noise kernel properties -->


## Method II: Multivariate Gaussian

**Theoretical Foundation**: This approach models displacement means as an multivariate Gaussian with a constraint-satisfying mean vector, ensuring incompressibility in expectation

**State Vector Definition**: Define the global mean displacement vector:
$$\boldsymbol{\mu} = \begin{pmatrix} \boldsymbol{\mu}_u \\ \boldsymbol{\mu}_v \end{pmatrix} \in \mathbb{R}^{2N}$$
where $\boldsymbol{\mu}_u = [\mu_{u,1,1,1}, \ldots, \mu_{u,N_x,N_y,N_z}]^T$ and $\boldsymbol{\mu}_v = [\mu_{v,1,1,1}, \ldots, \mu_{v,N_x,N_y,N_z}]^T$ with $N = N_x N_y N_z$ total grid points.

**Spatial Covariance Structure**: The covariance matrix has block structure:
$$\boldsymbol{\Sigma} = \begin{pmatrix} \sigma_u^2 \mathbf{K} & \rho_{uv}\sigma_u\sigma_v \mathbf{K} \\ \rho_{uv}\sigma_u\sigma_v \mathbf{K} & \sigma_v^2 \mathbf{K} \end{pmatrix}$$

The spatial correlation matrix $\mathbf{K} \in \mathbb{R}^{N \times N}$ is defined by:
$$[\mathbf{K}]_{(i,j,k),(i',j',k')} = \exp\left(-\frac{\|(i,j,k) - (i',j',k')\|^2}{2\ell^2}\right)$$
where $\ell > 0$ controls spatial correlation length.

**Constraint-Satisfying Mean Construction**: For interior grid points, construct mean vector $\boldsymbol{\mu}_0$ satisfying:
$$\frac{\mu_{u,i+1,j,k} - \mu_{u,i-1,j,k}}{2} + \frac{\mu_{v,i,j+1,k} - \mu_{v,i,j-1,k}}{2} = 0$$

for all $(i,j,k) \in \{2,\ldots,N_x-1\} \times \{2,\ldots,N_y-1\} \times \{1,\ldots,N_z\}$.

The constraint matrix $\mathbf{A} \in \mathbb{R}^{N_{\text{int}} \times 2N}$ encodes the discrete divergence operator, where $N_{\text{int}} = (N_x-2)(N_y-2)N_z$ is the number of interior points. For each interior point $(i,j,k)$, let $\mathbf{e}_{\ell}$ denote the standard basis column vector with 1 at position $\ell$ and 0 elsewhere. The constraint matrix row for interior point $(i,j,k)$ is:

$$[\mathbf{A}]_{(i,j,k)} = \frac{1}{2}\left(-\mathbf{e}_{\text{idx}(i-1,j,k)}^T + \mathbf{e}_{\text{idx}(i+1,j,k)}^T - \mathbf{e}_{N+\text{idx}(i,j-1,k)}^T + \mathbf{e}_{N+\text{idx}(i,j+1,k)}^T\right)$$

where $\text{idx}(\cdot)$ maps 3D coordinates to linear indices in $[1,N]$

Let $\mu_0$ denote the solution of the constraint system $\mathbf{A}\boldsymbol{\mu}_0 = \mathbf{0}$ 
<!-- 1. Setting boundary displacement means arbitrarily (e.g., $\boldsymbol{\mu}_{\text{boundary}} = \mathbf{0}$)
1. Solving the resulting linear system for interior means using flow conservation: Since the constraint matrix $\mathbf{A}$ has more variables than equations (the system is underdetermined), we can use the pseudoinverse or choose a particular solution. One approach is to use the minimum norm solution $\boldsymbol{\mu}_{\text{int}} = \mathbf{A}^+(\mathbf{0} - \mathbf{A}\boldsymbol{\mu}_{\text{boundary}})$ where $\mathbf{A}^+$ is the Moore-Penrose pseudoinverse, ensuring the interior displacement means satisfy the discrete incompressibility constraints while being consistent with the chosen boundary values. -->

**Sampling**: Generate field realizations via:
$$\boldsymbol{\mu} \sim \mathcal{N}(\boldsymbol{\mu}_0, \boldsymbol{\Sigma})$$

**Expected Incompressibility**: Since $\mathbb{E}[\boldsymbol{\mu}] = \boldsymbol{\mu}_0$ and $\mathbf{A}\boldsymbol{\mu}_0 = \mathbf{0}$:
$$\mathbb{E}[\nabla \cdot \mathbf{c}] = \mathbb{E}[\mathbf{A}\boldsymbol{\mu}] = \mathbf{A}\mathbb{E}[\boldsymbol{\mu}] = \mathbf{A}\boldsymbol{\mu}_0 = \mathbf{0}$$



<!--
# Appendix

## Sampling from Multivariate Gaussian with Squared Exponential Kernel over a Grid

### Problem Statement

Consider sampling from a multivariate Gaussian process $\mathbf{f} \sim \mathcal{GP}(0, k(\cdot, \cdot))$ evaluated on a regular grid $\{(i,j,k) : i=1,\ldots,N_x, j=1,\ldots,N_y, k=1,\ldots,N_z\}$ with squared exponential kernel:

$$k(\mathbf{x}, \mathbf{x}') = \sigma^2 \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2\ell^2}\right)$$

**Objective**: Efficiently sample $\mathbf{f} = [f(1,1,1), f(1,1,2), \ldots, f(N_x,N_y,N_z)]^T \in \mathbb{R}^N$ where $N = N_x N_y N_z$.

### Covariance Matrix Structure Analysis

**Grid Indexing**: Map 3D coordinates to linear indices via lexicographic ordering:
$$\text{idx}(i,j,k) = (i-1)N_y N_z + (j-1)N_z + k$$

**Covariance Matrix Definition**: The covariance matrix $\mathbf{K} \in \mathbb{R}^{N \times N}$ has entries:
$$[\mathbf{K}]_{pq} = k(\mathbf{x}_p, \mathbf{x}_q) = \sigma^2 \exp\left(-\frac{\|\mathbf{x}_p - \mathbf{x}_q\|^2}{2\ell^2}\right)$$

where $\mathbf{x}_p$ corresponds to the 3D coordinates of linear index $p$.

<!-- **Toeplitz Structure Analysis**: For a 1D grid with spacing $\Delta x$, the covariance matrix is **Toeplitz** since:
$$[\mathbf{K}]_{ij} = k(i\Delta x, j\Delta x) = \sigma^2 \exp\left(-\frac{(i-j)^2(\Delta x)^2}{2\ell^2}\right) = f(|i-j|)$$

However, for multi-dimensional grids, the covariance matrix is **not Toeplitz** due to the grid boundary effects and multi-dimensional distance structure.

**Block-Toeplitz Structure**: For 2D/3D grids, $\mathbf{K}$ exhibits a **Block-Toeplitz with Toeplitz Blocks (BTTB)** structure when periodic boundary conditions are assumed, enabling FFT-based computations. -->
<!--  
### Standard Sampling via Cholesky Decomposition

**Method**: Compute Cholesky decomposition $\mathbf{K} = \mathbf{L}\mathbf{L}^T$ and sample:
$$\mathbf{f} = \mathbf{L}\boldsymbol{\xi}, \quad \boldsymbol{\xi} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

**Computational Complexity**:
- Cholesky decomposition: $O(N^3) = O((N_x N_y N_z)^3)$
- Forward solve: $O(N^2)$
- **Total**: $O(N^3)$ - prohibitive for large grids

**Numerical Stability**: Add regularization $\mathbf{K} + \epsilon \mathbf{I}$ where $\epsilon \sim 10^{-6}$ to ensure positive definiteness.

### Circulant Embedding for Efficient Sampling

**Motivation**: Exploit FFT for $O(N \log N)$ sampling by embedding the covariance structure into a circulant matrix.

**Circulant Matrix Property**: A matrix $\mathbf{C} \in \mathbb{R}^{M \times M}$ is circulant if:
$$[\mathbf{C}]_{ij} = c_{(i-j) \bmod M}$$

**Key Result**: Circulant matrices are diagonalized by the DFT matrix:
$$\mathbf{C} = \mathbf{F}^H \boldsymbol{\Lambda} \mathbf{F}$$
where $\mathbf{F}$ is the DFT matrix and $\boldsymbol{\Lambda} = \text{diag}(\mathbf{F}\mathbf{c})$ with $\mathbf{c} = [c_0, c_1, \ldots, c_{M-1}]^T$.

### Periodic Extension Construction for 3D Grids

**3D Grid Extension**: For a 3D grid with dimensions $(N_x, N_y, N_z)$, extend to a periodic grid with dimensions $(2N_x, 2N_y, 2N_z)$. The circulant embedding requires constructing a Block-Circulant with Circulant Blocks (BCCB) structure.

**Index Notation**: 
- Original grid points: $(i, j, k)$ where $i \in \{0, 1, \ldots, N_x-1\}$, $j \in \{0, 1, \ldots, N_y-1\}$, $k \in \{0, 1, \ldots, N_z-1\}$
- Extended grid points: $(i', j', k')$ where $i' \in \{0, 1, \ldots, 2N_x-1\}$, $j' \in \{0, 1, \ldots, 2N_y-1\}$, $k' \in \{0, 1, \ldots, 2N_z-1\}$
- Grid spacing: $\Delta x$, $\Delta y$, $\Delta z$ in each dimension

**Wrapped Distance Function**: For any two points $(i_1, j_1, k_1)$ and $(i_2, j_2, k_2)$ on the extended grid, define the wrapped distance components:

$$d_x^{\text{wrap}} = \min(|i_1 - i_2| \cdot \Delta x, (2N_x - |i_1 - i_2|) \cdot \Delta x)$$
$$d_y^{\text{wrap}} = \min(|j_1 - j_2| \cdot \Delta y, (2N_y - |j_1 - j_2|) \cdot \Delta y)$$
$$d_z^{\text{wrap}} = \min(|k_1 - k_2| \cdot \Delta z, (2N_z - |k_1 - k_2|) \cdot \Delta z)$$

**Wrapped Distance Vector**: 
$$\mathbf{d}_{\text{wrap}} = [d_x^{\text{wrap}}, d_y^{\text{wrap}}, d_z^{\text{wrap}}]^T$$

**Extended Squared Exponential Kernel**: The periodic kernel is:
$$k_{\text{periodic}}((i_1, j_1, k_1), (i_2, j_2, k_2)) = \sigma^2 \exp\left(-\frac{\|\mathbf{d}_{\text{wrap}}\|^2}{2\ell^2}\right)$$

**Circulant Generator Array**: The 3D circulant structure is determined by the generator array $\mathbf{C} \in \mathbb{R}^{2N_x \times 2N_y \times 2N_z}$ where:
$$C_{i,j,k} = k_{\text{periodic}}((0,0,0), (i,j,k))$$

**Explicit Generator Formula**: For the squared exponential kernel:
$$C_{i,j,k} = \sigma^2 \exp\left(-\frac{(d_x^{(i)})^2 + (d_y^{(j)})^2 + (d_z^{(k)})^2}{2\ell^2}\right)$$

where:
- $d_x^{(i)} = \min(i \cdot \Delta x, (2N_x - i) \cdot \Delta x)$
- $d_y^{(j)} = \min(j \cdot \Delta y, (2N_y - j) \cdot \Delta y)$  
- $d_z^{(k)} = \min(k \cdot \Delta z, (2N_z - k) \cdot \Delta z)$

**Matrix Representation**: The full covariance matrix $\mathbf{K}_{\text{extended}} \in \mathbb{R}^{M \times M}$ with $M = 8N_xN_yN_z$ has entries:
$$[\mathbf{K}_{\text{extended}}]_{pq} = C_{i_p - i_q \bmod 2N_x, j_p - j_q \bmod 2N_y, k_p - k_q \bmod 2N_z}$$

where $(i_p, j_p, k_p)$ are the 3D coordinates corresponding to linear index $p$.

**Multi-dimensional Extension**: For a 3D grid, construct the extended grid with dimensions $(2N_x, 2N_y, 2N_z)$ and define:

$$k_{\text{periodic}}(\mathbf{i}, \mathbf{j}) = k(\mathbf{d}_{\text{wrap}})$$

where $\mathbf{d}_{\text{wrap}}$ is the wrapped distance:
$$[d_{\text{wrap}}]_\ell = \min(|i_\ell - j_\ell|, 2N_\ell - |i_\ell - j_\ell|)$$

### FFT-Based Sampling Algorithm

**Step 1: Eigenvalue Computation**
Compute the eigenvalues of the circulant matrix via FFT:
$$\boldsymbol{\lambda} = \text{FFT}(\mathbf{c})$$
where $\mathbf{c}$ is the first row of the circulant matrix.

**Step 2: Positivity Check**
Verify $\lambda_k \geq 0$ for all $k$. If negative eigenvalues exist due to numerical errors, set $\lambda_k := \max(0, \lambda_k)$ or increase regularization.

**Step 3: Sample Generation**
1. Generate complex Gaussian noise $\boldsymbol{\xi}$ where each component is $\xi_k = \xi_k^{(r)} + i\xi_k^{(i)}$ with $\xi_k^{(r)}, \xi_k^{(i)} \sim \mathcal{N}(0, 1)$ independently
2. Ensure Hermitian symmetry: $\xi_{-k} = \overline{\xi_k}$ (complex conjugate) to guarantee real output
3. Compute $\boldsymbol{\eta} = \sqrt{\boldsymbol{\lambda}} \odot \boldsymbol{\xi}$ (element-wise product)
4. Apply inverse FFT: $\mathbf{f}_{\text{extended}} = \text{IFFT}(\boldsymbol{\eta})$
5. Extract original grid values: $\mathbf{f} = \mathbf{f}_{\text{extended}}[1:N_x, 1:N_y, 1:N_z]$

**Step 4: Real-Valued Output**
The output $\mathbf{f}$ is automatically real-valued due to Hermitian symmetry. Any residual imaginary components (typically $\sim 10^{-16}$) are numerical artifacts and can be discarded via $\mathbf{f} = \text{Re}(\mathbf{f})$.

**Note on Complex Noise Necessity**: Complex Gaussian noise with Hermitian symmetry is essential because:
- The eigenvalues $\boldsymbol{\lambda}$ already possess Hermitian symmetry (being the FFT of real vector $\mathbf{c}$)
- To ensure $\boldsymbol{\eta} = \sqrt{\boldsymbol{\lambda}} \odot \boldsymbol{\xi}$ maintains this symmetry, $\boldsymbol{\xi}$ must also be Hermitian symmetric
- This guarantees IFFT produces purely real output, satisfying the requirement for real-valued Gaussian process samples

### Computational Complexity Analysis

**FFT-Based Method**:
- Eigenvalue computation: $O(M \log M)$ where $M = 8N_x N_y N_z$
- Sample generation: $O(M \log M)$
- **Total**: $O(N \log N)$ per sample

**Memory Requirements**:
- Standard Cholesky: $O(N^2)$ storage
- Circulant embedding: $O(M) = O(N)$ storage

**Scalability**: The FFT approach enables sampling on grids with $N \sim 10^6$ - $10^9$ points, compared to $N \sim 10^3$ - $10^4$ for direct Cholesky.

### Theoretical Guarantees

**Exactness Condition**: The circulant embedding produces exact samples from the target Gaussian process if the extended covariance matrix is positive semidefinite.

**Approximation Quality**: When negative eigenvalues are truncated, the sampling approximates the target distribution with error bounded by:
$$\|\mathbf{K}_{\text{true}} - \mathbf{K}_{\text{approx}}\|_F \leq \sqrt{\sum_{k: \lambda_k < 0} \lambda_k^2}$$

**Boundary Effects**: The periodic extension introduces artificial correlations across boundaries. For correlation length $\ell \ll \min(N_x, N_y, N_z)$, boundary effects are negligible in the interior of the domain.
 -->




<!-- **Spatial Smoothness**: The multi-octave construction naturally provides spatial smoothness through:
- Low-frequency octaves ($m=0,1$) contribute large-scale coherent structures
- High-frequency octaves ($m \geq 2$) add fine-scale detail
- Amplitude decay ($A_m \propto \beta^m$) ensures convergence and smoothness

**Resulting Field Properties**:
- **Computational efficiency**: $O(M \cdot N)$ scaling with grid size
- **Natural spatial correlation**: Controlled by base length scale $L$
- **Multi-scale structure**: Octave combination produces realistic turbulence-like patterns
- **Statistical control**: Mean displacement and correlation easily parameterized -->




