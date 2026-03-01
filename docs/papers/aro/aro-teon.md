# TARO: Tensorized Adaptively Rotated Optimization

**Combining Cross-Layer Tensor Structure (TEON) with Adaptive Rotation Policy (ARO) for LLM Pre-Training**

---

## 1. Motivation

Two recent advances in matrix-parameter optimization for LLMs address complementary weaknesses of MUON:

| Method | Key Idea | Limitation |
|--------|----------|------------|
| **TEON** | Stack gradients from K adjacent layers into a tensor, then orthogonalize the matricization — exploits cross-layer correlations | Uses fixed orthogonalization (Polar/NS), which is geometrically suboptimal |
| **ARO** | Replace orthogonalization with an adaptive rotation policy that maximizes instantaneous loss decrease | Operates layer-by-layer, missing cross-layer structure |

**TARO** unifies both: apply ARO's adaptive rotation on TEON's tensorized gradient matricizations, with parameter grouping justified by the transformer's rotational symmetry structure.

---

## 2. Mathematical Foundations

### 2.1 Notation

| Symbol | Meaning |
|--------|---------|
| $W^{(\ell)} \in \mathbb{R}^{m \times n}$ | Weight matrix of layer $\ell$ |
| $G^{(\ell)}_t$ | Gradient of $W^{(\ell)}$ at step $t$ |
| $M^{(\ell)}_t$ | Momentum (EMA of gradients) for layer $\ell$ |
| $K$ | Number of layers to stack (typically $K = 2$) |
| $\mathcal{M}_1(\cdot)$ | Mode-1 matricization of a 3-tensor |
| $R_t \in \mathbb{R}^{m \times m}$ | Adaptive rotation matrix (orthogonal) |
| $f_t(\cdot)$ | Base optimizer map (e.g. SinkGD) |
| $\eta$ | Learning rate |

### 2.2 TEON: Tensorized Gradient Structure

TEON stacks momenta from $K$ consecutive layers into a 3-tensor:

$$
\mathcal{T}_t = \text{Stack}\bigl(M^{(\ell)}_t,\; M^{(\ell+1)}_t,\; \dots,\; M^{(\ell+K-1)}_t\bigr) \;\in\; \mathbb{R}^{m \times n \times K}
$$

Mode-1 matricization unfolds along the row dimension:

$$
Z_t = \mathcal{M}_1(\mathcal{T}_t) \;\in\; \mathbb{R}^{m \times (nK)}
$$

This creates a wide matrix where columns encode both within-layer and across-layer gradient structure. TEON then orthogonalizes $Z_t$ via Newton-Schulz or Polar decomposition, which is equivalent to projecting onto the spectral norm ball.

**TEON convergence bound** (Theorem 1 from the paper):

$$
\min_{0 \le t < T} \| \nabla f(W_\tau) \|_* \;\le\; \sqrt{\frac{2L_{\text{TEON}} \cdot \Delta_0}{T}}
$$

where $L_{\text{TEON}} \le L_{\text{MUON}} \le K \cdot L_{\text{TEON}}$, meaning tensorization can improve the effective smoothness constant by up to $\sqrt{K}\times$.

### 2.3 ARO: Adaptive Rotation Policy

ARO formulates gradient preconditioning as **rotated steepest descent**:

$$
\Delta W_t = -\eta \, R_t \, f_t\!\bigl(R_t^\top G_t\bigr)
$$

where $R_t$ is an orthogonal rotation matrix chosen to maximize instantaneous loss decrease:

$$
\mathcal{J}(R;\, M,\, f) = \bigl\langle M,\; R \, f(R^\top M) \bigr\rangle
$$

**Key insight:** Eigen-rotation (used by SOAP, Muon, SPlus) sets $R_t = U_t$ from $\text{SVD}(M_t) = U_t \Sigma_t V_t^\top$. This minimizes the *variance* of the alignment score but also minimizes its *expectation*. ARO finds a better tradeoff.

**ARO rotation update rule:**

$$
R_t = \text{QR}\!\Bigl( M_t \cdot f_t\!\bigl(R_{t-1}^\top M_t\bigr)^\top \Bigr)
$$

This is a one-step gradient ascent on $\mathcal{J}$ with respect to $R$, using QR factorization to project back onto the orthogonal group.

### 2.4 SinkGD: Robust Stateless Base Optimizer

ARO's recommended base optimizer $f_t$ is **SinkGD** — iterative alternating row/column normalization:

$$
f_{\text{sink}}(X) = \text{Sink}^{(N)}(X)
$$

where one Sinkhorn iteration is:

$$
\text{Sink}(X): \quad X \leftarrow \text{RowNorm}(X), \quad X \leftarrow \text{ColNorm}(X)
$$

with $\text{RowNorm}(X)_i = X_i / \|X_i\|_2$ and $\text{ColNorm}(X)_j = X_j / \|X_j\|_2$.

Typical $N = 5$ iterations. SinkGD is stateless (no second moment), handles extreme aspect ratios well, and is more robust than SignGD/RowNorm for rectangular matrices.

---

## 3. TARO: The Unified Algorithm

### 3.1 Core Idea

Replace TEON's orthogonalization step with ARO's adaptive rotation, applied to the tensorized matricization:

$$
\boxed{
\Delta Z_t = -\eta \, R_t \, f_{\text{sink}}\!\bigl(R_t^\top Z_t\bigr)
}
$$

where $Z_t = \mathcal{M}_1(\mathcal{T}_t) \in \mathbb{R}^{m \times (nK)}$ is the tensorized momentum and $R_t \in \mathbb{R}^{m \times m}$.

### 3.2 Symmetry-Grounded Parameter Grouping

TEON empirically finds that stacking QKV matrices works best but lacks theoretical justification for *why*. ARO's symmetry hypothesis (Section 6) provides exactly this:

**Theorem (Rotational Symmetry).** For a transformer with RMSNorm pre-normalization, the loss is invariant under a shared rotation of all residual-stream matrices:

$$
\mathcal{L}(R \cdot W_{\text{all}}) = \mathcal{L}(W_{\text{all}}) \quad \forall\, R \in \text{SO}(m)
$$

where $W_{\text{all}}$ includes $W_Q, W_K, W_V$ (all layers), embeddings, and the LM head.

**Implication for grouping:** All matrices that transform the residual stream from the left share the same rotational symmetry. We should group parameters by their *symmetry position*, not just by layer type:

- **Group A** (residual-stream left-multiplied): $W_Q^{(\ell)}, W_K^{(\ell)}, W_V^{(\ell)}$ — stack across layers
- **Group B** (residual-stream right-multiplied): $W_O^{(\ell)}, W_{\text{up}}^{(\ell)}$ — stack across layers  
- **Group C** (special): Embeddings, LM head — couple via chain rule

Within each group, a single rotation $R_t$ is shared across all $K$ stacked layers. This is computationally cheaper (one QR per group vs one per layer) and theoretically grounded.

### 3.3 Rotation Update on Tensorized Gradients

The rotation is updated using the ARO policy applied to the wide matricization:

$$
R_t = \text{CholQR}\!\Bigl( Z_t \cdot f_{\text{sink}}\!\bigl(R_{t-1}^\top Z_t\bigr)^\top \Bigr)
$$

where $\text{CholQR}(A) = A \cdot (A^\top A)^{-1/2}$ is shifted Cholesky QR for numerical stability.

**Why CholQR over standard QR?** The matrix $Z_t \cdot f_{\text{sink}}(\cdot)^\top$ is $m \times m$ and well-conditioned when $nK \gg m$, making Cholesky QR both faster and more stable than Householder QR.

### 3.4 Inverse Matricization → Per-Layer Updates

After computing $\Delta Z_t \in \mathbb{R}^{m \times (nK)}$, reshape back to individual layer updates:

$$
\Delta Z_t = \bigl[\Delta W^{(\ell)}_t \;\big|\; \Delta W^{(\ell+1)}_t \;\big|\; \cdots \;\big|\; \Delta W^{(\ell+K-1)}_t\bigr]
$$

where each $\Delta W^{(k)}_t \in \mathbb{R}^{m \times n}$ is the $k$-th horizontal slice.

### 3.5 LR Scaling and Normalization

Following both papers, apply RMS normalization for hyperparameter transfer from AdamW:

$$
\Delta W^{(k)}_t \leftarrow \Delta W^{(k)}_t \cdot \frac{\text{RMS}(\Delta W^{(k)}_{\text{AdamW}})}{\text{RMS}(\Delta W^{(k)}_t)}
$$

In practice, this simplifies to scaling by $\sqrt{m \cdot n} / \|\Delta W^{(k)}_t\|_F$ with an appropriate LR multiplier (typically $0.02 \times \eta_{\text{AdamW}}$).

---

## 4. Algorithm: TARO

### 4.1 Initialization

```
INPUT:
  model parameters W = {W^(ℓ)} for all layers ℓ
  learning rate η, momentum coefficient β (default 0.95)
  stacking depth K (default 2)
  Sinkhorn iterations N_sink (default 5)
  warmup steps T_warmup
  
INIT:
  M^(ℓ) ← 0   for all ℓ        // momentum buffers
  R_g ← I_m    for each group g  // rotation matrices (identity)
  
DEFINE GROUPS by symmetry:
  Group_QKV  = {W_Q^(ℓ), W_K^(ℓ), W_V^(ℓ)}  stacked in blocks of K adjacent layers
  Group_O    = {W_O^(ℓ)}                       stacked in blocks of K adjacent layers
  Group_MLP  = {W_up^(ℓ), W_gate^(ℓ)}         stacked in blocks of K adjacent layers
  Group_DOWN = {W_down^(ℓ)}                    stacked in blocks of K adjacent layers
  
  // Embeddings and LM head: use AdamW (following both papers' convention)
```

### 4.2 Main Loop

```
FOR t = 1, 2, ..., T:

  // ─── Step 1: Compute gradients ───
  G^(ℓ)_t ← ∇_{W^(ℓ)} L(W_t)   for all ℓ

  // ─── Step 2: Update momenta (momentum-first) ───
  FOR each layer ℓ:
    M^(ℓ)_t ← β · M^(ℓ)_{t-1} + (1 - β) · G^(ℓ)_t

  // ─── Step 3: Process each symmetry group ───
  FOR each group g with rotation R_g:
    
    // 3a. Tensorize: stack K consecutive layer momenta
    FOR each block b of K adjacent layers in group g:
      T_t ← Stack(M^(ℓ_b)_t, M^(ℓ_b+1)_t, ..., M^(ℓ_b+K-1)_t)   // ∈ R^{m × n × K}
      Z_t ← Mode1_Matricize(T_t)                                    // ∈ R^{m × (nK)}
      
      // 3b. SinkGD on rotated gradient
      Y_t ← R_g^⊤ · Z_t                        // rotate into alignment basis
      S_t ← Sinkhorn(Y_t, N_sink)               // alternating row/col norm
      
      // 3c. Update rotation (ARO policy)
      P_t ← Z_t · S_t^⊤                        // ∈ R^{m × m}
      R_g ← CholQR(P_t)                         // project back to O(m)
      
      // 3d. Compute update
      ΔZ_t ← R_g · S_t                          // rotated normalized gradient
      
      // 3e. Inverse matricize → per-layer updates
      [ΔW^(ℓ_b)_t, ..., ΔW^(ℓ_b+K-1)_t] ← Split(ΔZ_t, K)
      
      // 3f. RMS normalization per layer
      FOR each layer k in block b:
        ΔW^(k)_t ← ΔW^(k)_t · (√(m·n) / ‖ΔW^(k)_t‖_F)

  // ─── Step 4: Apply updates with warmup ───
  η_t ← η · min(1, t / T_warmup)
  FOR each layer ℓ:
    W^(ℓ)_{t+1} ← W^(ℓ)_t - η_t · ΔW^(ℓ)_t
```

### 4.3 Subroutines

```
FUNCTION Sinkhorn(X, N):
  // Alternating row/column normalization
  FOR i = 1, ..., N:
    X ← RowNorm(X)      // X[i,:] ← X[i,:] / ‖X[i,:]‖₂
    X ← ColNorm(X)      // X[:,j] ← X[:,j] / ‖X[:,j]‖₂
  RETURN X

FUNCTION CholQR(A):
  // Numerically stable orthogonalization via shifted Cholesky
  // A ∈ R^{m × m}, assumed well-conditioned
  C ← A^⊤ A + ε · I    // shift for stability, ε ≈ 1e-6
  L ← Cholesky(C)       // C = L L^⊤
  Q ← A · L^{-⊤}       // solve triangular system
  RETURN Q

FUNCTION Mode1_Matricize(T):
  // T ∈ R^{m × n × K} → Z ∈ R^{m × (nK)}
  // Concatenate along columns: [T[:,:,1] | T[:,:,2] | ... | T[:,:,K]]
  RETURN HorizontalConcat(T[:,:,1], T[:,:,2], ..., T[:,:,K])
```

---

## 5. Theoretical Justification

### 5.1 Why Tensorization + Rotation > Either Alone

**Claim.** Let $L_{\text{TARO}}$ be the effective smoothness constant of TARO. Then:

$$
L_{\text{TARO}} \;\le\; L_{\text{ARO}} \;\le\; L_{\text{MUON}} \;\le\; K \cdot L_{\text{TARO}}
$$

**Argument sketch:**

1. **Tensorization reduces smoothness** (from TEON, Theorem 1): Stacking $K$ layers and jointly processing the matricization yields an effective smoothness constant $L_{\text{TEON}} \le L_{\text{MUON}}$ because the spectral norm of the combined Hessian block is bounded by the individual blocks.

2. **ARO rotation improves over orthogonalization** (from ARO, Proposition 2): For the same gradient matrix, ARO's rotation achieves $\mathcal{J}(R_{\text{ARO}}) \ge \mathcal{J}(R_{\text{eigen}})$ — strictly greater instantaneous loss decrease than eigen-rotation (which subsumes orthogonalization).

3. **Combined:** Applied to the tensorized matricization, ARO rotation exploits both the reduced smoothness (from cross-layer structure) and the improved descent direction (from adaptive rotation).

### 5.2 Why Symmetry Justifies Cross-Layer Stacking

The rotational symmetry $\mathcal{L}(RW) = \mathcal{L}(W)$ for $R \in \text{SO}(m)$ implies:

1. The loss landscape has a **continuous symmetry orbit** — a manifold of equivalent solutions
2. All residual-stream matrices within this orbit share the **same optimal rotation** $R^*$
3. Stacking them into a tensor and applying a single rotation is not an approximation — it is **exact** under this symmetry

This explains TEON's empirical finding that QKV stacking works (same symmetry position) while MLP mixing hurts (different symmetry positions for up-projection vs down-projection).

### 5.3 Computational Cost Analysis

| Operation | MUON (per layer) | TARO (per K-layer block) | Savings |
|-----------|-------------------|--------------------------|---------|
| Orthogonalization / Rotation | $K \times O(m^2 n)$ | $1 \times O(m^2 \cdot nK)$ | Same FLOPS |
| Rotation matrix update | — | $O(m^2 \cdot nK) + O(m^3)$ | New cost |
| QR / CholQR | $K \times O(m^2 n)$ | $1 \times O(m^3)$ | $\approx K\times$ cheaper |
| Memory: rotation state | 0 | $m^2$ per group | Negligible |

**Net:** TARO adds $O(m^3)$ per block for rotation update but saves $O(K \cdot m^2 n)$ by sharing one rotation across $K$ layers. For typical LLM dimensions ($n \gg m$ or $n \approx m$), the rotation update is dominated by the Sinkhorn step.

---

## 6. Hyperparameter Recommendations

Based on both papers' ablations:

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| $K$ (stacking depth) | 2 | Diminishing returns for K > 2; K=2 optimal in TEON |
| $\beta$ (momentum) | 0.95 | Standard for MUON-family; momentum-first design |
| $N_{\text{sink}}$ (Sinkhorn iters) | 5 | Sufficient convergence; minimal cost |
| $\eta$ (learning rate) | $0.02 \times \eta_{\text{AdamW}}$ | With RMS normalization |
| $\varepsilon$ (CholQR shift) | $10^{-6}$ | Numerical stability |
| $T_{\text{warmup}}$ | 5% of total steps | Linear warmup |
| Weight decay | Same as AdamW | Applied directly to $W$, not through optimizer |
| Rotation init | $R_0 = I$ | Cold-start; converges within ~100 steps |

### 6.1 Parameter Group LR Multipliers

| Group | LR Multiplier | Optimizer |
|-------|---------------|-----------|
| QKV projections | $1.0 \times$ | TARO |
| Output projections | $1.0 \times$ | TARO |
| MLP up/gate | $1.0 \times$ | TARO |
| MLP down | $0.5 \times$ | TARO (or AdamW fallback) |
| Embeddings | $1.0 \times$ | AdamW |
| LM head | $0.5 \times$ | AdamW |
| LayerNorm / biases | $1.0 \times$ | AdamW |

### 6.2 Stacking Strategy

```
Given L total layers, K=2:

QKV blocks:  [(Q₁,K₁,V₁, Q₂,K₂,V₂), (Q₃,K₃,V₃, Q₄,K₄,V₄), ...]
  → Each block: stack 2 layers' QKV, matricize to R^{d_model × (d_head·n_heads·3·2)}

O blocks:    [(O₁, O₂), (O₃, O₄), ...]
  → Each block: R^{d_model × (d_model·2)}

MLP blocks:  [(Up₁,Gate₁, Up₂,Gate₂), (Up₃,Gate₃, Up₄,Gate₄), ...]
  → Each block: R^{d_model × (d_ff·2·2)}

If L is odd: last layer processed solo (K=1 fallback = standard ARO)
```

---

## 7. Key Differences from Pure TEON / Pure ARO

| Aspect | TEON | ARO | TARO |
|--------|------|-----|------|
| Cross-layer coupling | ✅ Tensor stacking | ❌ Layer-wise | ✅ Tensor stacking |
| Rotation policy | Fixed (Polar/NS) | Adaptive (ARO rule) | Adaptive (ARO rule) |
| Base optimizer | Identity (just ortho) | SinkGD | SinkGD |
| MLP support | ⚠️ QKV only (stability) | ✅ All matrices | ✅ All matrices (SinkGD stable) |
| Theoretical basis for grouping | Empirical | Symmetry hypothesis | Symmetry hypothesis |
| Rotation sharing | Implicit (one ortho per tensor) | Explicit (chain-coupled) | Explicit (per symmetry group) |
| Extra state per group | None | $R \in \mathbb{R}^{m \times m}$ | $R \in \mathbb{R}^{m \times m}$ |

---

## 8. Expected Gains (Predicted)

Based on the individual papers' reported speedups over AdamW:

- **TEON over MUON:** ~5-10% improvement (from cross-layer structure)
- **ARO over MUON:** ~10-15% improvement (from adaptive rotation)
- **TARO (predicted):** ~15-25% over MUON, ~1.4-1.5× over AdamW

These are **multiplicative** improvements on orthogonal axes (geometry of rotation vs scope of information), so the combined effect should be at least partially additive.

---

## 9. Implementation Checklist

- [ ] Momentum buffers for all matrix parameters
- [ ] Parameter grouping by symmetry position
- [ ] Tensor stacking + mode-1 matricization per group block
- [ ] Sinkhorn normalization (5 iterations)
- [ ] ARO rotation update with CholQR
- [ ] Inverse matricization to per-layer updates
- [ ] RMS normalization per layer
- [ ] AdamW fallback for embeddings, LM head, 1D params
- [ ] Linear LR warmup
- [ ] Rotation matrices initialized to identity
- [ ] Distributed training: rotation matrices on same device as parameter shard

---

*References:*
- *TEON: Tensorized Orthonormalization Beyond Layer-Wise MUON (arXiv:2601.23261v2)*
- *ARO: Adaptively Rotated Optimization (arXiv:2602.09006v1)*