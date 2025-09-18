## Smoothness of Multivariate Gaussian
**Lipschitz Continuity of Gaussian Processes with Squared Exponential Kernel**

**Definition**: A function $f: \mathbb{R}^d \to \mathbb{R}$ is Lipschitz continuous with constant $L$ if:
$$|f(\mathbf{x}) - f(\mathbf{x}')| \leq L \|\mathbf{x} - \mathbf{x}'\|_2$$
for all $\mathbf{x}, \mathbf{x}' \in \mathbb{R}^d$.

**Setup**: Let $f(\mathbf{x}) \sim \mathcal{GP}(0, k(\mathbf{x}, \mathbf{x}'))$ with squared exponential kernel:
$$k(\mathbf{x}, \mathbf{x}') = \sigma^2 \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2\ell^2}\right)$$

**Step 1: Compute Partial Derivatives of Kernel**

First partial derivative:
$$\frac{\partial k(\mathbf{x}, \mathbf{x}')}{\partial x_i} = \sigma^2 \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2\ell^2}\right) \cdot \left(-\frac{2(x_i - x'_i)}{2\ell^2}\right) = -\frac{\sigma^2(x_i - x'_i)}{\ell^2} \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2\ell^2}\right)$$

Second mixed partial derivative:
$$\frac{\partial^2 k(\mathbf{x}, \mathbf{x}')}{\partial x_i \partial x'_i} = \frac{\partial}{\partial x'_i}\left[-\frac{\sigma^2(x_i - x'_i)}{\ell^2} \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2\ell^2}\right)\right]$$

$$= -\frac{\sigma^2}{\ell^2}\left[-\exp\left(-\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2\ell^2}\right) + (x_i - x'_i)\frac{\partial}{\partial x'_i}\exp\left(-\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2\ell^2}\right)\right]$$

$$= -\frac{\sigma^2}{\ell^2}\left[-\exp\left(-\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2\ell^2}\right) + (x_i - x'_i)\left(\frac{(x_i - x'_i)}{\ell^2} \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2\ell^2}\right)\right)\right]$$

$$= \frac{\sigma^2}{\ell^2} \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2\ell^2}\right)\left[1 - \frac{(x_i - x'_i)^2}{\ell^2}\right]$$

**Step 2: Derivative Process Covariance**

The partial derivative $\frac{\partial f}{\partial x_i}$ is a Gaussian process with covariance kernel:
$$k_{i,i}(\mathbf{x}, \mathbf{x}') = \frac{\partial^2 k(\mathbf{x}, \mathbf{x}')}{\partial x_i \partial x'_i} = \frac{\sigma^2}{\ell^2}\left[1 - \frac{(x_i - x'_i)^2}{\ell^2}\right]\exp\left(-\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2\ell^2}\right)$$

**Step 3: Variance of Derivative Process**

At any point $\mathbf{x}$, the variance of the $i$-th partial derivative is:
$$\text{Var}\left(\frac{\partial f(\mathbf{x})}{\partial x_i}\right) = k_{i,i}(\mathbf{x}, \mathbf{x}) = \frac{\sigma^2}{\ell^2}$$

**Step 4: Gaussian Tail Bound**

For a zero-mean Gaussian random variable $X$ with variance $\sigma^2$:
$$\mathbb{P}[|X| > t] \leq 2\exp\left(-\frac{t^2}{2\sigma^2}\right)$$

Setting $\mathbb{P}[|X| > t] = \delta$ and solving for $t$:
$$t = \sigma\sqrt{2\log(2/\delta)}$$

**Step 5: Union Bound Over Derivatives**
For all $d$ partial derivatives simultaneously, we apply the union bound. Let $A_i$ be the event that $\left|\frac{\partial f(\mathbf{x})}{\partial x_i}\right| > \frac{\sigma}{\ell}\sqrt{2\log(2d/\delta)}$. Then:

$$\mathbb{P}\left[\bigcup_{i=1}^d A_i\right] \leq \sum_{i=1}^d \mathbb{P}[A_i] = \sum_{i=1}^d \frac{\delta}{d} = \delta$$

Therefore, with probability $1 - \delta$:
$$\left|\frac{\partial f(\mathbf{x})}{\partial x_i}\right| \leq \frac{\sigma}{\ell}\sqrt{2\log(2d/\delta)}$$
for all $i \in \{1,\ldots,d\}$ simultaneously.

**Step 6: Gradient Norm Bound**

The Euclidean norm of the gradient satisfies:
$$\|\nabla f(\mathbf{x})\| = \sqrt{\sum_{i=1}^d \left(\frac{\partial f(\mathbf{x})}{\partial x_i}\right)^2} \leq \sqrt{d} \max_{i} \left|\frac{\partial f(\mathbf{x})}{\partial x_i}\right|$$

Therefore, with probability $1 - \delta$:
$$\|\nabla f(\mathbf{x})\| \leq \sqrt{d} \cdot \frac{\sigma}{\ell}\sqrt{2\log(2d/\delta)}$$

**Step 7: Mean Value Theorem**

By the mean value theorem, for any $\mathbf{x}, \mathbf{x}' \in \mathbb{R}^d$, there exists $\mathbf{c}$ on the line segment between $\mathbf{x}$ and $\mathbf{x}'$ such that:
$$f(\mathbf{x}) - f(\mathbf{x}') = \nabla f(\mathbf{c})^T (\mathbf{x} - \mathbf{x}')$$

**Step 8: Lipschitz Constant**

Taking absolute values and using Cauchy-Schwarz inequality:
$$|f(\mathbf{x}) - f(\mathbf{x}')| = |\nabla f(\mathbf{c})^T (\mathbf{x} - \mathbf{x}')| \leq \|\nabla f(\mathbf{c})\| \|\mathbf{x} - \mathbf{x}'\|$$

**Conclusion**: With probability $1 - \delta$, the GP sample paths are Lipschitz continuous with constant:
$$L = \sqrt{2d\log(2d/\delta)} \cdot \frac{\sigma}{\ell}$$



### Mathematical Foundation of Simplex Noise

**Definition**: Simplex noise $\eta(\mathbf{x})$ is a gradient noise function that produces pseudo-random values with controlled spatial correlation properties. Unlike Perlin noise which uses hypercubes, simplex noise operates on simplexes (triangles in 2D, tetrahedra in 3D), reducing computational complexity.

**Core Construction**:
1. **Lattice Structure**: The space is divided into a regular simplex lattice. For 3D, this creates a tetrahedral grid where each point $\mathbf{x}$ falls within a specific tetrahedron.

2. **Gradient Assignment**: Each lattice vertex $\mathbf{v}_i$ is assigned a pseudo-random unit gradient vector $\mathbf{g}_i$ based on a hash function of its coordinates:
   $$\mathbf{g}_i = \text{hash}(\mathbf{v}_i) \mapsto \mathbf{u} \in \mathbb{S}^{d-1}$$

3. **Influence Functions**: For a point $\mathbf{x}$ within a simplex, each vertex contributes through a smooth kernel:
   $$\eta(\mathbf{x}) = \sum_{i} w_i(\mathbf{x}) \cdot (\mathbf{g}_i \cdot (\mathbf{x} - \mathbf{v}_i))$$
   
   where $w_i(\mathbf{x})$ is a smooth weight function with compact support:
   $$w_i(\mathbf{x}) = \max(0, r^2 - \|\mathbf{x} - \mathbf{v}_i\|^2)^k$$
   
   with $r$ being the influence radius and $k \geq 2$ ensuring $C^{k-1}$ continuity.

**Continuity Properties**: The simplex noise function $\eta(\mathbf{x})$ exhibits:
- **$C^{k-1}$ Smoothness**: Determined by the kernel exponent $k$ in the weight function
- **Bounded Derivatives**: The gradient magnitude is bounded by the lattice structure
- **Local Lipschitz Continuity**: Within each simplex cell, the function is Lipschitz continuous

### Multi-Octave Structure and Fractal Properties

**Octave Motivation**: Single-scale noise produces uniform texture lacking natural complexity. Multi-octave combination creates **fractal-like** patterns with structure at multiple scales, mimicking natural phenomena like turbulence, terrain, and atmospheric flows.

**Mathematical Basis - Fractional Brownian Motion**: The multi-octave construction approximates fractional Brownian motion (fBm) with Hurst parameter $H$:

$$B_H(t) = \int_{-\infty}^{\infty} \frac{e^{ixt} - 1}{|x|^{H + 1/2}} dW(x)$$

**Discrete Approximation**: The multi-octave noise:
$$f(\mathbf{x}) = \sum_{m=0}^{M-1} A_m \eta(2^m \mathbf{x} / L)$$

approximates fBm when:
- **Amplitude scaling**: $A_m = A_0 \beta^m$ with $\beta = 2^{-H}$
- **Frequency doubling**: Each octave has twice the spatial frequency
- **Persistence parameter**: $\beta \in (0,1)$ controls roughness ($\beta \to 0$: smoother, $\beta \to 1$: rougher)

**Spectral Properties**: The power spectral density follows:
$$S(k) \propto k^{-(2H+1)}$$

For atmospheric turbulence, typical values are $H \in [0.3, 0.7]$, corresponding to $\beta \in [0.62, 0.81]$.

### Continuity Analysis of Multi-Octave Fields

**Individual Octave Smoothness**: Each octave $\eta(2^m \mathbf{x} / L)$ inherits the $C^{k-1}$ smoothness of the base simplex noise, but scaled:

**Gradient Bound for Octave $m$**: 
$$\left\|\nabla \eta(2^m \mathbf{x} / L)\right\| \leq C \cdot \frac{2^m}{L}$$

where $C$ is the Lipschitz constant of the base noise function.

**Multi-Octave Gradient**: The gradient of the combined field is:
$$\nabla f(\mathbf{x}) = \sum_{m=0}^{M-1} A_m \nabla \eta(2^m \mathbf{x} / L) = \sum_{m=0}^{M-1} A_m \frac{2^m}{L} \nabla \eta(2^m \mathbf{x} / L)$$

**Convergence Analysis**: The series converges when:
$$\sum_{m=0}^{\infty} A_m \frac{2^m}{L} = \frac{A_0}{L} \sum_{m=0}^{\infty} \beta^m 2^m = \frac{A_0}{L} \sum_{m=0}^{\infty} (2\beta)^m$$

This converges if $2\beta < 1$, i.e., $\beta < 0.5$. For $\beta \geq 0.5$, the field becomes non-differentiable (fractal-like).

**Practical Smoothness**: With finite $M$ octaves and $\beta < 1$, the field is $C^{k-1}$ smooth with Lipschitz constant:
$$L_{\text{field}} = \frac{A_0 C}{L} \sum_{m=0}^{M-1} (2\beta)^m = \frac{A_0 C}{L} \cdot \frac{1-(2\beta)^M}{1-2\beta}$$

**Field Correlation Structure**: The two-point correlation function exhibits power-law decay:
$$\mathbb{E}[f(\mathbf{x})f(\mathbf{x} + \mathbf{h})] \approx \sigma_{\text{field}}^2 \|\mathbf{h}\|^{2H-2}$$

for $\|\mathbf{h}\| \gg L$, where $H = -\log_2(\beta)$ is the Hurst parameter.

## Alternative Approaches
- **Probability re-allocation or convex combination for defining similar pmf's**
- \notes{1. maybe perform similarity analysis betweeen the constructed fields and the ERA5 grid
}


## Circulant Matrices: A Complete Tutorial

### Definition and Structure

A **circulant matrix** is a special type of matrix where each row is a cyclic shift of the previous row. For a vector $\mathbf{c} = [c_0, c_1, \ldots, c_{n-1}]^T$, the corresponding $n \times n$ circulant matrix $\mathbf{C}$ is:

$$\mathbf{C} = \begin{bmatrix}
c_0 & c_{n-1} & c_{n-2} & \cdots & c_1 \\
c_1 & c_0 & c_{n-1} & \cdots & c_2 \\
c_2 & c_1 & c_0 & \cdots & c_3 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
c_{n-1} & c_{n-2} & c_{n-3} & \cdots & c_0
\end{bmatrix}$$

**Key Property**: The matrix is completely determined by its first row (or equivalently, first column).

### Circular Convolution Connection

Circulant matrices naturally arise from **circular convolution**. If $\mathbf{y} = \mathbf{C}\mathbf{x}$, then:
$$y_k = \sum_{j=0}^{n-1} c_{(k-j) \bmod n} x_j$$

This is precisely the definition of circular convolution: $\mathbf{y} = \mathbf{c} \circledast \mathbf{x}$.

**Insight**: Matrix-vector multiplication with a circulant matrix is equivalent to circular convolution.

**Circulant Matrix Property**: A matrix $\mathbf{C} \in \mathbb{R}^{M \times M}$ is circulant if:
$$[\mathbf{C}]_{ij} = c_{(i-j) \bmod M}$$

<!-- **Key Result**: Circulant matrices are diagonalized by the DFT matrix:
$$\mathbf{C} = \mathbf{F}^H \boldsymbol{\Lambda} \mathbf{F}$$
where $\mathbf{F}$ is the DFT matrix and $\boldsymbol{\Lambda} = \text{diag}(\mathbf{F}\mathbf{c})$ with $\mathbf{c} = [c_0, c_1, \ldots, c_{M-1}]^T$. -->

### DFT and Eigenvalue Derivation

**Theorem**: The eigenvectors of any circulant matrix are the columns of the DFT matrix, and eigenvalues are given by the DFT of the first row.

**Proof**: Let $\omega_n = e^{-2\pi i/n}$ be the primitive $n$-th root of unity. The DFT matrix has entries:
$$[\mathbf{F}]_{jk} = \frac{1}{\sqrt{n}} \omega_n^{jk}$$

For the $k$-th eigenvector $\mathbf{v}_k = \frac{1}{\sqrt{n}}[1, \omega_n^k, \omega_n^{2k}, \ldots, \omega_n^{(n-1)k}]^T$:

$$[\mathbf{C}\mathbf{v}_k]_j = \sum_{i=0}^{n-1} c_{(j-i) \bmod n} \frac{1}{\sqrt{n}} \omega_n^{ik}$$

Substituting $m = (j-i) \bmod n$:
$$[\mathbf{C}\mathbf{v}_k]_j = \frac{1}{\sqrt{n}} \sum_{m=0}^{n-1} c_m \omega_n^{(j-m)k} = \frac{\omega_n^{jk}}{\sqrt{n}} \sum_{m=0}^{n-1} c_m \omega_n^{-mk}$$

The sum $\sum_{m=0}^{n-1} c_m \omega_n^{-mk}$ is precisely $\sqrt{n} \cdot \text{DFT}(\mathbf{c})_k = \sqrt{n} \lambda_k$.

Therefore: $\mathbf{C}\mathbf{v}_k = \lambda_k \mathbf{v}_k$ where $\lambda_k = \text{DFT}(\mathbf{c})_k$.

### Step 3: Spectral Decomposition

Since circulant matrices are **normal** (they commute with their conjugate transpose), they admit spectral decomposition:
$$\mathbf{C} = \mathbf{F}^* \boldsymbol{\Lambda} \mathbf{F}$$

where:
- $\mathbf{F}$ is the DFT matrix
- $\boldsymbol{\Lambda} = \text{diag}(\lambda_0, \lambda_1, \ldots, \lambda_{n-1})$
- $\lambda_k = \text{DFT}(\mathbf{c})_k$

### Step 4: Fast Matrix Operations

**Matrix-Vector Multiplication**: 
$$\mathbf{C}\mathbf{x} = \mathbf{F}^* (\boldsymbol{\Lambda} (\mathbf{F}\mathbf{x})) = \text{IFFT}(\boldsymbol{\lambda} \odot \text{FFT}(\mathbf{x}))$$
**Complexity**: $O(n \log n)$ instead of $O(n^2)$

**Matrix Powers**:
$$\mathbf{C}^p = \mathbf{F}^* \boldsymbol{\Lambda}^p \mathbf{F}$$
where $\boldsymbol{\Lambda}^p = \text{diag}(\lambda_0^p, \lambda_1^p, \ldots, \lambda_{n-1}^p)$

**Matrix Functions**:
$$f(\mathbf{C}) = \mathbf{F}^* \text{diag}(f(\lambda_0), f(\lambda_1), \ldots, f(\lambda_{n-1})) \mathbf{F}$$

### Step 5: Block Circulant Extension

For multi-dimensional problems, we use **block circulant matrices**. A 2D circulant matrix for an $N_x \times N_y$ grid has the form:

$$\mathbf{C}_{2D} = \begin{bmatrix}
\mathbf{C}_0 & \mathbf{C}_{N_y-1} & \cdots & \mathbf{C}_1 \\
\mathbf{C}_1 & \mathbf{C}_0 & \cdots & \mathbf{C}_2 \\
\vdots & \vdots & \ddots & \vdots \\
\mathbf{C}_{N_y-1} & \mathbf{C}_{N_y-2} & \cdots & \mathbf{C}_0
\end{bmatrix}$$

where each $\mathbf{C}_j$ is itself an $N_x \times N_x$ circulant matrix.

**Key Result**: Eigenvalues are given by the 2D DFT:
$$\lambda_{k_1,k_2} = \text{DFT}_{2D}(\mathbf{c})_{k_1,k_2}$$

### Step 6: Application to Covariance Matrices

**Circulant Embedding for GPs**: Given a covariance function $K(x,y)$ on a regular grid, we construct a circulant matrix by:

1. **Extend the grid**: Create a $2N \times 2N$ extended grid (in 1D)
2. **Define circulant entries**: Set $c_k = K(0, k \Delta x)$ for the periodic extension
3. **Ensure positive definiteness**: The covariance function must satisfy certain conditions

**Why this works**: 
- Stationarity of the GP ensures the covariance depends only on $|x-y|$
- Circulant structure naturally encodes this translation invariance
- FFT diagonalization enables $O(N \log N)$ sampling

**Hermitian Symmetry**: For real-valued processes, the circulant matrix must be Hermitian, requiring:
$$c_k = c_{n-k}^* \quad \text{(complex conjugate symmetry)}$$

This ensures eigenvalues $\lambda_k = \text{DFT}(\mathbf{c})_k$ are real, guaranteeing positive semidefiniteness.

### Understanding Hermitian Symmetry in Detail

**Why Hermitian Symmetry is Required**: 

1. **Real-valued outputs**: Since we want to sample real-valued Gaussian fields, the covariance matrix must be real and symmetric.

2. **Circulant constraint**: In a circulant matrix, element $(i,j)$ depends only on $(i-j) \bmod n$. For the matrix to be real and symmetric, we need:
   $$[\mathbf{C}]_{ij} = [\mathbf{C}]_{ji}$$

3. **Translation to first row**: This symmetry constraint translates to the first row (which defines the entire circulant matrix) as:
   $$c_k = c_{n-k} \quad \text{for } k = 1, 2, \ldots, n-1$$

**Concrete Example**: For $n=6$, the first row must satisfy:
- $c_1 = c_5$ (elements 1 and 5 positions apart)
- $c_2 = c_4$ (elements 2 and 4 positions apart)  
- $c_3 = c_3$ (middle element, automatically satisfied)
- $c_0$ (diagonal element, no constraint)

So a valid first row might be: $[c_0, c_1, c_2, c_3, c_2, c_1]$

**DFT Eigenvalue Consequences**:
When the circulant matrix has Hermitian symmetry, its eigenvalues (computed via DFT) are guaranteed to be real:
$$\lambda_k = \sum_{j=0}^{n-1} c_j e^{-2\pi i jk/n}$$

The Hermitian symmetry $c_j = c_{n-j}$ ensures that:
$$\lambda_k = c_0 + 2\sum_{j=1}^{\lfloor n/2 \rfloor} c_j \cos(2\pi jk/n) + c_{n/2}\cos(\pi k) \quad \text{(if n is even)}$$

This is purely real, which is essential for:
- **Positive semidefiniteness**: All $\lambda_k \geq 0$
- **Real sampling**: $\sqrt{\lambda_k}$ is real
- **Computational stability**: No complex arithmetic in sampling

**Practical Implementation**: When constructing circulant embedding for covariance functions, always ensure the extended covariance vector satisfies this symmetry before computing the DFT eigenvalues.


## Sampling from Multivariate Gaussian via Circulant Embedding

**Problem Setup**: Sample $\mathbf{f} \sim \mathcal{N}(\boldsymbol{\mu}, \mathbf{K})$ where $\mathbf{K}$ is the covariance matrix of a stationary Gaussian process on a grid.

### Step-by-Step Derivation

**Step 1: Circulant Extension**

**Assumptions on Covariance Structure**: We assume $\mathbf{K}$ is the covariance matrix of a stationary Gaussian process on a regular 1D grid with spacing $\Delta x$. Specifically:
- Grid points: $x_i = i \Delta x$ for $i = 0, 1, \ldots, N-1$
- Covariance function: $K(x, y) = K(|x - y|)$ depends only on distance
- Matrix elements: $[\mathbf{K}]_{ij} = K(|x_i - x_j|) = K(|i - j| \Delta x)$
  
**Circulant Extension Construction**: Given the original $N \times N$ covariance matrix $\mathbf{K}$, we construct a circulant matrix $\mathbf{C}$ of size $M \times M$ with $M = 2N-1$ such that:
$$\mathbf{K} = \mathbf{C}[0:N-1, 0:N-1]$$

where $\mathbf{C}[0:N-1, 0:N-1]$ denotes the top-left $N \times N$ submatrix of $\mathbf{C}$, i.e., the submatrix containing rows 0 through $N-1$ and columns 0 through $N-1$ of the larger $M \times M$ circulant matrix $\mathbf{C}$.

**General Toeplitz to Circulant Embedding**: For any symmetric Toeplitz matrix $\mathbf{T} \in \mathbb{R}^{N \times N}$ with entries $[\mathbf{T}]_{ij} = t_{|i-j|}$, we can construct a circulant matrix $\mathbf{C} \in \mathbb{R}^{M \times M}$ with $M = 2N-1$ that contains $\mathbf{T}$ as its top-left block.

**Theorem (Toeplitz Circulant Embedding)**: Let $\mathbf{T} \in \mathbb{R}^{N \times N}$ be a symmetric Toeplitz matrix with elements $[\mathbf{T}]_{ij} = t_{|i-j|}$ where $t_k$ are the Toeplitz values for lags $k = 0, 1, \ldots, N-1$. Define a circulant matrix $\mathbf{C} \in \mathbb{R}^{M \times M}$ with $M = 2N-1$ by its first row:

$$c_k = \begin{cases}
t_k & \text{for } k = 0, 1, \ldots, N-1 \\
t_{M-k} & \text{for } k = N, N+1, \ldots, M-1
\end{cases}$$

Then $[\mathbf{C}]_{ij} = [\mathbf{T}]_{ij}$ for all $i,j \in \{0, 1, \ldots, N-1\}$.

**Proof**: We need to show that for any $i,j \in \{0, 1, \ldots, N-1\}$, the circulant property $[\mathbf{C}]_{ij} = c_{(i-j) \bmod M}$ yields the correct Toeplitz value $t_{|i-j|}$.

**Case 1**: $i \geq j$, so $i - j \geq 0$ and $|i-j| = i-j$.
- Since $0 \leq i-j \leq N-1$, we have $(i-j) \bmod M = i-j$
- Therefore: $[\mathbf{C}]_{ij} = c_{i-j} = t_{i-j} = t_{|i-j|} = [\mathbf{T}]_{ij}$

**Case 2**: $i < j$, so $i - j < 0$ and $|i-j| = j-i$.
- We have $i - j \in \{-(N-1), -(N-2), \ldots, -1\}$
- Therefore: $(i-j) \bmod M = M + (i-j) = M - (j-i)$
- Since $j-i \in \{1, 2, \ldots, N-1\}$, we have $M-(j-i) \in \{M-1, M-2, \ldots, M-(N-1)\} = \{N, N+1, \ldots, M-1\}$
- By our construction: $c_{M-(j-i)} = t_{M-(M-(j-i))} = t_{j-i}$
- Therefore: $[\mathbf{C}]_{ij} = c_{M-(j-i)} = t_{j-i} = t_{|i-j|} = [\mathbf{T}]_{ij}$

<!-- **Why $M = 2N-1$ is Sufficient**: The choice $M = 2N-1$ is the minimal size required because:
- We need indices $k \in \{0, 1, \ldots, N-1\}$ for the first $N$ elements of the circulant first row
- We need indices $k \in \{N, N+1, \ldots, M-1\}$ for the remaining elements, which must satisfy $M-k \in \{1, 2, \ldots, N-1\}$
- This gives us $k \in \{M-(N-1), M-(N-2), \ldots, M-1\} = \{N, N+1, \ldots, M-1\}$
- The minimal $M$ satisfying both constraints is $M = N + (N-1) = 2N-1$ -->

**First Row Construction for Covariance Matrices**: For our specific case where $t_k = K(k \Delta x)$ represents covariance values, the circulant first row becomes:
$$c_k = \begin{cases}
K(k \Delta x) & \text{for } k = 0, 1, \ldots, N-1 \\
K((M-k) \Delta x) & \text{for } k = N, N+1, \ldots, M-1
\end{cases}$$

**Symmetry Verification**: The construction automatically ensures $c_k = c_{M-k}$ for $k = 1, 2, \ldots, M-1$:
- For $k \in \{1, \ldots, N-1\}$: $c_k = K(k \Delta x)$ and $c_{M-k} = K((M-(M-k)) \Delta x) = K(k \Delta x)$
- For $k \in \{N, \ldots, M-1\}$: $c_k = K((M-k) \Delta x)$ and $c_{M-k} = K((M-k) \Delta x)$

This symmetry property ensures that the circulant matrix $\mathbf{C}$ is symmetric, guaranteeing all eigenvalues are real.

This construction ensures:
1. **Embedding property**: $[\mathbf{C}]_{ij} = K(|i-j| \Delta x) = [\mathbf{K}]_{ij}$ for $i,j \in \{0, 1, \ldots, N-1\}$
2. **Symmetry property**: The construction ensures $c_k = c_{M-k}$ for all valid indices, making the circulant matrix symmetric and ensuring real eigenvalues


**Step 2: Eigendecomposition via DFT**
Since $\mathbf{C}$ is circulant, it can be diagonalized as:
$$\mathbf{C} = \mathbf{F}^H \boldsymbol{\Lambda} \mathbf{F}$$

where:
- $\mathbf{F}$ is the DFT matrix: $[\mathbf{F}]_{jk} = \frac{1}{\sqrt{M}} e^{-2\pi i jk/M}$
- $\boldsymbol{\Lambda} = \text{diag}(\boldsymbol{\lambda})$ with eigenvalues $\boldsymbol{\lambda} = \text{DFT}(\mathbf{c})$

**Step 3: Matrix Square Root**
The matrix square root is computed in the spectral domain:
$$\mathbf{C}^{1/2} = \mathbf{F}^H \boldsymbol{\Lambda}^{1/2} \mathbf{F}$$

where $\boldsymbol{\Lambda}^{1/2} = \text{diag}(\sqrt{\lambda_0}, \sqrt{\lambda_1}, \ldots, \sqrt{\lambda_{M-1}})$.

**Step 4: Generate Standard Gaussian Random Vector**

**Hermitian Symmetry Definition**: A complex vector $\boldsymbol{\xi} \in \mathbb{C}^M$ has Hermitian symmetry if $\xi_{M-k} = \xi_k^*$ for $k = 1, \ldots, \lfloor M/2 \rfloor$, with $\xi_0$ and $\xi_{M/2}$ (if $M$ even) being real.

**FFT Real Output Property**: The inverse FFT of a Hermitian symmetric vector produces a real-valued output. Conversely, the FFT of a real vector produces a Hermitian symmetric result.

**Eigenvalue Reality**: Since the circulant first row $\mathbf{c}$ is symmetric ($c_k = c_{M-k}$), all eigenvalues $\lambda_k = \text{DFT}(\mathbf{c})_k$ are real. For the eigenvalues to be non-negative (ensuring $\mathbf{C}$ is positive semidefinite), the embedded covariance function must satisfy additional conditions - specifically, the circulant extension must preserve the positive definiteness of the original covariance matrix.

**\question{what happens when C is not positive semidefinite?}**

This complex construction ensures $\text{IFFT}(\sqrt{\boldsymbol{\lambda}} \odot \text{FFT}(\boldsymbol{\xi}))$ produces real output while handling the non-PSD case gracefully.

**Step 5: Transform to Target Distribution (Cholesky Equivalent)**

**Standard Gaussian Process Sampling**: The conventional approach for sampling from a multivariate Gaussian $\mathbf{f} \sim \mathcal{N}(\boldsymbol{\mu}, \mathbf{K})$ relies on the **Cholesky decomposition**:

1. **Cholesky factorization**: Decompose the covariance matrix as $\mathbf{K} = \mathbf{L}\mathbf{L}^T$, where $\mathbf{L}$ is a lower triangular matrix
2. **Sample generation**: Generate $\boldsymbol{\xi} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and compute $\mathbf{f} = \mathbf{L}\boldsymbol{\xi} + \boldsymbol{\mu}$

**Why Cholesky Works**: If $\boldsymbol{\xi} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, then $\mathbf{L}\boldsymbol{\xi}$ has covariance:
$$\text{Cov}(\mathbf{L}\boldsymbol{\xi}) = \mathbf{L} \text{Cov}(\boldsymbol{\xi}) \mathbf{L}^T = \mathbf{L} \mathbf{I} \mathbf{L}^T = \mathbf{L}\mathbf{L}^T = \mathbf{K}$$

**Circulant Embedding as Matrix Square Root**: In our method, $\mathbf{C}^{1/2}$ plays the role of the Cholesky factor $\mathbf{L}$. The circulant matrix $\mathbf{C}$ is symmetric by construction (due to the Hermitian symmetry property $c_k = c_{M-k}$ proven above). To prove $\mathbf{C}$ is positive semidefinite, we use the spectral characterization:



**Step 6: Establish matrix square root**
Since all eigenvalues are non-negative, $\mathbf{C}$ is positive semidefinite and admits a unique positive semidefinite square root $\mathbf{C}^{1/2}$ such that:
$$\mathbf{C} = \mathbf{C}^{1/2} (\mathbf{C}^{1/2})^T$$

This square root can be computed in the spectral domain as $\mathbf{C}^{1/2} = \mathbf{F}^H \text{diag}(\sqrt{\lambda_0}, \sqrt{\lambda_1}, \ldots, \sqrt{\lambda_{M-1}}) \mathbf{F}$.

The matrix square root $\mathbf{C}^{1/2}$ is computed efficiently in the spectral domain:

$$\boldsymbol{\eta} = \mathbf{C}^{1/2} \boldsymbol{\xi} = \mathbf{F}^H (\sqrt{\boldsymbol{\lambda}} \odot \mathbf{F}\boldsymbol{\xi})$$

where $\odot$ denotes element-wise multiplication. This spectral computation of $\mathbf{C}^{1/2}$ avoids the $O(M^3)$ cost of standard Cholesky decomposition, achieving the same result in $O(M \log M)$ operations.

**Step 6: Extract Original Sample**
The desired sample is:
$$\mathbf{f} = \boldsymbol{\eta}[1:N] + \boldsymbol{\mu}$$

**Step 7: Computational Implementation**
In practice, this becomes:
1. Compute eigenvalues: $\boldsymbol{\lambda} = \text{FFT}(\mathbf{c})$
2. Generate Hermitian symmetric noise: $\boldsymbol{\xi}$ as above
3. Transform noise: $\tilde{\boldsymbol{\xi}} = \text{FFT}(\boldsymbol{\xi})$
4. Scale by square root: $\boldsymbol{\eta}_{\text{freq}} = \sqrt{\boldsymbol{\lambda}} \odot \tilde{\boldsymbol{\xi}}$
5. Transform back: $\boldsymbol{\eta} = \text{IFFT}(\boldsymbol{\eta}_{\text{freq}})$
6. Extract and add mean: $\mathbf{f} = \text{Re}(\boldsymbol{\eta}[1:N]) + \boldsymbol{\mu}$

**Mathematical Verification**:
The covariance of the generated sample is:
$$\text{Cov}(\mathbf{f}) = \mathbb{E}[\boldsymbol{\eta}[1:N] \boldsymbol{\eta}[1:N]^T] = \mathbf{C}[1:N, 1:N] = \mathbf{K}$$

This confirms that $\mathbf{f} \sim \mathcal{N}(\boldsymbol{\mu}, \mathbf{K})$ as desired.

### Computational Advantages Summary

1. **Storage**: $O(n)$ instead of $O(n^2)$ - only store the first row
2. **Matrix-vector products**: $O(n \log n)$ via FFT instead of $O(n^2)$
3. **Eigendecomposition**: Available analytically via DFT
4. **Matrix functions**: Computed directly in spectral domain
5. **Sampling**: Square root and random multiplication in $O(n \log n)$

This mathematical structure is what makes the circulant embedding method so powerful for large-scale Gaussian process sampling.


<!-- **Proof of Positive Semidefiniteness**: A symmetric matrix is positive semidefinite if and only if all its eigenvalues are non-negative. Since $\mathbf{C}$ is circulant, its eigenvalues are given by $\lambda_k = \text{DFT}(\mathbf{c})_k$ for $k = 0, 1, \ldots, M-1$.

**Step 1: Express eigenvalues in terms of the covariance function**
The DFT of the circulant vector $\mathbf{c}$ can be expressed as:
$$\lambda_k = \sum_{j=0}^{M-1} c_j e^{-2\pi i jk/M}$$

By the construction of $\mathbf{c}$, we have $c_j = K(d_j)$ where $d_j$ is the distance corresponding to index $j$ in the periodic extension. Specifically:
- For $j = 0, 1, \ldots, N-1$: $d_j = j \Delta x$
- For $j = N, N+1, \ldots, M-1$: $d_j = (M-j) \Delta x$

Therefore:
$$\lambda_k = \sum_{j=0}^{M-1} K(d_j) e^{-2\pi i jk/M}$$

**Step 2: Reformulate as quadratic form**
We can rewrite the eigenvalue as:
$$\lambda_k = \sum_{j=0}^{M-1} \sum_{l=0}^{M-1} \delta_{jl} K(d_j) e^{-2\pi i jk/M}$$

where $\delta_{jl}$ is the Kronecker delta. This can be expressed as a quadratic form:
$$\lambda_k = \sum_{j=0}^{M-1} \sum_{l=0}^{M-1} w_j^* K(|d_j - d_l|) w_l$$

where $w_j = e^{-2\pi i jk/M}$ and we use the fact that $K(d_j) = K(|d_j - d_0|)$ since $d_0 = 0$.

**Step 3: Apply positive definiteness of covariance function**
Since $K(\cdot)$ is a valid covariance function, it is positive semidefinite. This means that for any finite set of points $\{x_0, x_1, \ldots, x_{M-1}\}$ and any complex weights $\{w_0, w_1, \ldots, w_{M-1}\}$:
$$\sum_{i,j=0}^{M-1} w_i^* K(|x_i - x_j|) w_j \geq 0$$

**Step 4: Map to our specific case**
In our case, we have points $\{x_j = j \Delta x \bmod (M \Delta x)\}_{j=0}^{M-1}$ on the periodic domain, and weights $w_j = e^{-2\pi i jk/M}$. The distance function in the periodic domain gives us $|x_i - x_j| = d_{|i-j|_M}$ where $|\cdot|_M$ denotes modular arithmetic.

However, due to the circulant structure and our specific construction of $\mathbf{c}$, we have:
$$\lambda_k = \sum_{j=0}^{M-1} c_j e^{-2\pi i jk/M} = \mathbf{w}_k^* \mathbf{K}_{\text{periodic}} \mathbf{w}_k$$

where $\mathbf{K}_{\text{periodic}}$ is the $M \times M$ periodic covariance matrix with entries $[\mathbf{K}_{\text{periodic}}]_{ij} = K(d_{|i-j|_M})$ and $\mathbf{w}_k = [1, e^{-2\pi i k/M}, e^{-2\pi i 2k/M}, \ldots, e^{-2\pi i (M-1)k/M}]^T$.

**Step 5: Conclude non-negativity**
Since $\mathbf{K}_{\text{periodic}}$ is a covariance matrix constructed from a valid covariance function on a periodic domain, it is positive semidefinite. Therefore:
$$\lambda_k = \mathbf{w}_k^* \mathbf{K}_{\text{periodic}} \mathbf{w}_k \geq 0$$

for all $k = 0, 1, \ldots, M-1$. -->