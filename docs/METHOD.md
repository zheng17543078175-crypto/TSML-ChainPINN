# Method and implementation

## Coordinate-to-field mapping

Input: normalized `(x,y,z,t)`. Each coordinate is standardized using fitting-subset population moments. Flow order is **[p,u,v,w]**; phase output is a liquid-fraction logit. A sigmoid gives alpha_L, with alpha_v = 1-alpha_L.

Fourier features implement Eq. (11): concatenate cosine and sine of `2*pi*B*xi`, using 64 log-spaced bands from 1 to 256 along all four coordinates. No raw-coordinate concatenation is used. Ten 512-wide residual blocks per branch implement Eq. (12), including the outer Softplus activation.

Branch-local self-attention precedes simultaneous bidirectional cross-attention. Both updates read pre-coupling features. Linear and attention weights use Xavier initialization.

**Attention semantics:** each coordinate is one batch element containing one feature token, retained from the archived architecture. Softmax has length one. This is pointwise feature exchange, not attention across mesh points. Predictions are independent of inference chunking, permitting pointwise coordinate derivatives. The paper does not specify another tokenization.

## Progressive optimization

| Stage | Trainable modules | Objective |
|---|---|---|
| 1 | Flow branch | Scaled flow L1 + 0.10 divergence + 0.05 momentum |
| 2 | Phase branch | Focal + 0.5 masked L1 + 2.0 false-positive loss |
| 3, epochs 1–20 | Phase branch | 14.0 × hybrid phase loss |
| 3, epochs 21–160 | All modules | Flow + 14.0 phase + 0.05 divergence + 0.05 momentum + 0.10 vapor |

Frozen modules remain in forward computation; gradients pass through them to trainable inputs, but frozen parameters do not update. Cross-attention parameters unfreeze only during joint optimization.

Focal target: `alpha_L < 0.90`; cavitation probability: `sigmoid(-logit_L)`; gamma=2. No extra class-balancing coefficient. Regression averages over cavitating points; the one-sided false-positive penalty averages over the complement. Empty subsets contribute zero.

The sampler draws 20% from the cavitation pool and 80% from the full snapshot. It does not guarantee exactly 20% cavitating points in the final batch. One fitting snapshot is chosen uniformly for each optimizer update.

## Dimensional physics

Derivatives are divided by coordinate normalization scales. Outputs return to SI units before forming residuals. Pressure supplied to ZGB must be absolute.

```text
rho_m = alpha_L*rho_L + (1-alpha_L)*rho_v
mu_m  = alpha_L*mu_L  + (1-alpha_L)*mu_v

R_m = d(rho_m*u)/dt + div(rho_m*u tensor u)
      + grad(p) - div(mu_m*(grad(u) + grad(u)^T))

R_v = rho_v*(d(alpha_v)/dt + u dot grad(alpha_v) + alpha_v*div(u))
      - m_vap + m_cond
```

Conservative products are differentiated without dropping variable-density terms. Stage 1 uses CFD phase values/gradients and fitting-only temporal differences. A local first-order lift supplies coefficient derivatives at sampled points. Stage 3 differentiates predicted phase.

ZGB uses positive parts `p_v-p` and `p-p_v`: exact ReLU for diagnosis, pressure-dimensional Softplus for training. Residual scales are Uref/Lref, rho_L Uref²/Lref and rho_v Uref/Lref. Divergence/momentum use squared residuals; vapor uses Huber loss. No unreported turbulence/eddy-viscosity closure is added to the neural loss.
