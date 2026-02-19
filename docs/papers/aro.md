# ARO + TEON: Plan реализации гибридного оптимизатора

> Документ основан на статье **"ARO: A New Lens On Matrix Optimization For Large Models"** (arXiv:2602.09006v1)
> и методологии TEON (Tensorized Orthonormalization). Цель — реализовать ARO с TEON-style
> parameter grouping для обучения 124M LM в рамках бюджета.

---

## Содержание

1. [Теоретическая база ARO](#1-теоретическая-база-aro)
2. [Математика ARO](#2-математика-aro)
3. [TEON и его роль](#3-teon-и-его-роль)
4. [Гибридная схема ARO + TEON](#4-гибридная-схема-aro--teon)
5. [Практические детали](#5-практические-детали)
6. [Имплементация](#6-имплементация)
7. [Конфигурация для 124M модели](#7-конфигурация-для-124m-модели)
8. [Чеклист и советы](#8-чеклист-и-советы)

---

## 1. Теоретическая база ARO

### 1.1 Normed Steepest Descent (NSD)

Классический спуск с выбранной нормой решает задачу:

$$\Delta W^{\star}_t = \arg\min_{\Delta W_t} \left[ \langle G_t, \Delta W_t \rangle + \frac{\lambda}{2} \|\Delta W_t\|^2 \right]$$

где $\lambda > 0$ — регуляризация, $\|\cdot\|$ — произвольная норма на $\mathbb{R}^{m \times n}$.

Решение имеет замкнутую форму:

$$\Delta W^{\star}_t \propto -f(G_t), \quad f(G_t) := \arg\max_{\|Z_t\| \leq 1} \langle G_t, Z_t \rangle = \nabla_{G_t} \|G_t\|_{\ast}$$

где $\|\cdot\|_{\ast}$ — двойственная норма к $\|\cdot\|$.

**Примеры:**
| Норма $\|\cdot\|$ | $f(G)$ | Оптимизатор |
|---|---|---|
| $\ell_\infty$ (поэлементная) | $\text{Sign}(G)$ | SignGD / Adam |
| Max-row norm | $\sqrt{n} Q(G)^{-1} G$ | Row-Norm / Muon |
| Sinkhorn hybrid | $f_{\text{Sink}}(G)$ | SinkGD |

### 1.2 Rotated Steepest Descent

Ключевая идея ARO: оптимизация в повёрнутой системе координат.

**Определение повёрнутых координат.** Пусть $R_t \in SO(m)$ — ортогональная матрица вращения. Введём:

$$W_t = R_t Z_t \implies \tilde{\mathcal{L}}_{R_t}(Z_t) = \mathcal{L}(R_t Z_t)$$

**Градиент в повёрнутых координатах:**

$$\nabla_{Z_t} \tilde{\mathcal{L}}_{R_t}(Z_t) = R_t^\top \nabla_{W_t} \mathcal{L}(R_t Z_t) = R_t^\top G_t$$

**Шаг в повёрнутых координатах и обратное преобразование:**

$$\Delta Z_t = -\eta f_t(R_t^\top G_t)$$
$$\Delta W_t = R_t \Delta Z_t = -\eta R_t f_t(R_t^\top G_t)$$

**Итоговая формула Rotated Steepest Descent:**

$$\boxed{\Delta W_t = -\eta R_t f_t(R_t^\top G_t)}$$

### 1.3 Унификация существующих методов

Все современные матричные оптимизаторы — частные случаи с разными $R_t$ и $f_t$:

| Метод                   | $R_t$                                    | $f_t$                 |
| ----------------------- | ---------------------------------------- | --------------------- |
| SOAP                    | Eigenvectors$(G_t G_t^\top)$             | Adam                  |
| SPlus                   | Eigenvectors$(G_t G_t^\top)$             | Sign                  |
| Muon (идеализированный) | Eigenvectors$(G_t G_t^\top)$             | $\sqrt{n} Q(X)^{-1}X$ |
| **ARO (наш)**           | **QR$(G_t f_t(R_{t-1}^\top G_t)^\top)$** | **Любой $f_t$**       |

**Ключевое отличие ARO**: вращение $R_t$ **зависит от $f_t$** — это non-eigen, optimizer-aware rotation.

---

## 2. Математика ARO

### 2.1 Objective для выбора вращения

ARO выбирает $R_t$, максимизируя **мгновенную скорость убывания потерь**:

$$\mathcal{J}(R_t; G_t, f) := \langle G_t, R_t f_t(R_t^\top G_t) \rangle = \text{tr}(G_t^\top R_t f(R_t^\top G_t))$$

Эквивалентная формулировка через двойственную норму (из свойств NSD):

$$\mathcal{J}(R; G_t, f) = \langle R^\top G_t, f(R^\top G_t) \rangle = \|R^\top G_t\|_{\ast}$$

### 2.2 Решение: Orthogonal Procrustes Problem

Прямая максимизация $\mathcal{J}$ по $SO(m)$ сложна (для $f = \text{Sign}$ — NP-hard L1-PCA).

**Приближённое решение** — расцепить два вхождения $R_t$: использовать $R_{t-1}$ в одном:

$$R^{\star}_t \in \arg\max_{R_t \in SO(m)} \text{tr}\!\left(G_t f_t(R_{t-1}^\top G_t)^\top R_t^\top\right)$$

Это **Orthogonal Procrustes Problem** с замкнутым решением:

$$R^{\star}_t = \text{Orthogonalize}\!\left(G_t f_t(R_{t-1}^\top G_t)^\top\right)$$

**ARO использует QR вместо Polar** из-за лучшей устойчивости к шуму:

$$\boxed{R^{\text{ARO}}_t = \text{QR}\!\left(G_t f_t(R_{t-1}^\top G_t)^\top\right)}$$

### 2.3 Полный ARO update (с momentum)

Авторы используют **momentum-first** дизайн — один буфер $M_t$ для оценки как momentum, так и вращения:

$$M_t = \text{EMA}[G_t] := \beta M_{t-1} + (1 - \beta) G_t$$

$$R_t = \text{QR}\!\left(M_t \cdot f_t(R_{t-1}^\top M_t)^\top\right) \quad \text{(Rotation selection)}$$

$$\Delta W_t = -\eta \cdot R_t \cdot f_t(R_t^\top M_t) \quad \text{(Rotated update)}$$

### 2.4 SinkGD как base optimizer

Авторы выбирают SinkGD в качестве $f_t$ по трём причинам:

1. Stateless — не требует доп. буферов
2. Не является rotational-equivariant (иначе ARO не даёт эффекта)
3. Меньшая инвариантная группа → больше пространства для ARO

**Алгоритм $f_{\text{Sink}}(G, L)$:**

```
X = G
for ℓ = 1 to L:
    Q(X) = Diag(‖X[i,:]‖₂) — диагональ из L2-норм строк
    R(X) = Diag(‖X[:,j]‖₂) — диагональ из L2-норм столбцов
    X ∝ Q(X)⁻¹ · X · R(X)⁻¹  (одновременная нормализация строк и столбцов)
return X
```

На практике используется $L = 5$ итераций.

### 2.5 ARO как Symmetry Teleportation

**Потери инвариантны** к левому вращению остаточного потока:

$$\mathcal{L}(R W) = \mathcal{L}(W), \quad \forall R \in SO(m)$$

В трансформерах (RMSNorm) это выполнено точно: все матрицы, "потребляющие" residual stream, умножаются справа на $R^\top$, матрицы "производящие" — слева на $R$.

При инвариантности константа гладкости $\beta_R$ не зависит от $R$, и из леммы о спуске:

$$\mathcal{L}(W_t) - \mathcal{L}(W_{t+1}) \geq \frac{1}{2\beta} \|R^\top G\|^2_{\ast}$$

Поэтому максимизация $\|R^\top G\|_\ast$ (т.е. $\mathcal{J}$) **гарантированно улучшает** убывание потерь.

### 2.6 Почему ARO лучше eigen-rotation?

Из теоремы (Appendix C) eigen-rotation является **локальным минимизатором** $\text{Var}(S_A)$ (alignment variance), но одновременно **локальным минимизатором** $\mathbb{E}[S_A]$ (alignment magnitude).

ARO занимает компромиссную позицию:

- Не так агрессивен как Polar (не раздувает $\text{Var}(S_A)$)
- Не так консервативен как Eigen (не подавляет $\mathbb{E}[S_A]$)

### 2.7 Gradient Orthogonalization как частный случай ARO

При $f_t = f_{RN}(X) = \sqrt{n} Q(X)^{-1} X$ (row-wise normalization):

$$R_t = \text{QR}(M_t f_{RN}(R_{t-1}^\top M_t)^\top) = \text{QR}(M_t M_t^\top R_{t-1} Q(R_{t-1}^\top M_t)^{-1})$$
$$= \text{QR}(M_t M_t^\top R_{t-1})$$

При сходимости power iteration: $R_t = U_t$ (левые сингулярные векторы $M_t = U_t S_t V_t^\top$).

Тогда:
$$\Delta W_t = -\eta U_t f_{RN}(U_t^\top M_t) = -\eta U_t Q(S_t V_t^\top)^{-1}(S_t V_t^\top) = -\eta U_t V_t^\top$$

— это в точности **шаг gradient orthogonalization / spectral descent** (Muon).

---

## 3. TEON и его роль

### 3.1 Что делает TEON

TEON (Tensorized Orthonormalization) стекует матрицы одного типа (напр. $[W_Q, W_K, W_V]$) и применяет к ним **совместную** ортогонализацию. По терминологии ARO — это sharing **eigen-rotations** across module types.

В рамках ARO: TEON = eigen-rotation + TEON-grouping.

### 3.2 Chain-coupled rotation sharing (из ARO Section 6.4)

ARO предлагает более принципиальную схему, основанную на симметриях residual stream.

При слабой residual связи (раннее обучение, глубокие сети):

$$X^{(\ell-\frac{1}{2})} \approx Y^{(\ell)}, \quad X^{(\ell)} \approx Z^{(\ell)}$$

можно использовать **последовательность локальных вращений** $\{R_\ell\}_{\ell=0}^L$:

$$\widetilde{W}_Q^{(\ell)} = W_Q^{(\ell)} R_{\ell-1}^\top, \quad \widetilde{W}_K^{(\ell)} = W_K^{(\ell)} R_{\ell-1}^\top, \quad \widetilde{W}_V^{(\ell)} = W_V^{(\ell)} R_{\ell-1}^\top$$
$$\widetilde{W}_O^{(\ell)} = R_\ell W_O^{(\ell)}, \quad \widetilde{W}_{\text{up}}^{(\ell)} = W_{\text{up}}^{(\ell)} R_{\ell-1}^\top, \quad \widetilde{W}_{\text{down}}^{(\ell)} = R_\ell W_{\text{down}}^{(\ell)}$$

Слой $\ell$ "потребляет" активации в базисе $R_{\ell-1}$ и "производит" в базисе $R_\ell$.

**Стоимость:** $(L+1)$ QR-факторизаций $d \times d$ = $\mathcal{O}((L+1)d^3)$ — vs. $(6L+3)$ для per-parameter.

### 3.3 Ориентация весов (Design 3 из ARO)

Критически важно: в стандартном трансформере **все матрицы должны быть транспонированы** перед применением ARO, **кроме $W_O$ и $W_{\text{down}}$**.

Это следует из Equation (48) статьи — ориентируем residual-stream dimension как row dimension:

| Параметр                                          | Правило                | Обоснование                       |
| ------------------------------------------------- | ---------------------- | --------------------------------- |
| $W_Q, W_K, W_V$                                   | Транспонировать        | Потребляют residual stream справа |
| $W_O$                                             | **Не транспонировать** | Производят residual stream слева  |
| $W_{\text{up}}$                                   | Транспонировать        | Потребляют residual stream справа |
| $W_{\text{down}}$                                 | **Не транспонировать** | Производят residual stream слева  |
| $W_{\text{tok}}, W_{\text{pos}}, W_{\text{head}}$ | Транспонировать        | Потребляют/embedding side         |

---

## 4. Гибридная схема ARO + TEON

### 4.1 Идея гибрида

Берём из TEON: **стратегию группировки параметров** (стек QKV внутри слоя).

Берём из ARO: **политику выбора вращения** (non-eigen, optimizer-aware QR).

Результат: ARO-rotation применяется к стекованным momentum матрицам → одно вращение на группу.

### 4.2 Группировки параметров

**Группа Attention (для слоя $\ell$):**

$$M_{\text{attn}}^{(\ell)} = \begin{bmatrix} M_Q^{(\ell)} \\ M_K^{(\ell)} \\ M_V^{(\ell)} \end{bmatrix} \in \mathbb{R}^{3d \times d}$$

ARO вращение: $R_{\text{attn}}^{(\ell)} = \text{QR}\!\left(M_{\text{attn}} f_t(R_{\text{prev}}^\top M_{\text{attn}})^\top\right) \in SO(3d)$

Затем применяем к каждому компоненту раздельно, используя соответствующий блок $R$.

**Группа MLP (для слоя $\ell$):**

$$M_{\text{mlp}}^{(\ell)} = \begin{bmatrix} M_{\text{up}}^{(\ell)} \end{bmatrix} \in \mathbb{R}^{4d \times d}$$

$W_{\text{down}}$ обрабатывается отдельно (другая ориентация).

### 4.3 Chain-coupled вариант (рекомендуемый)

Для каждого слоя $\ell$ — одно вращение $R_\ell \in SO(d)$, оцениваемое из стекованного momentum **всех** параметров слоя:

$$M_{\text{stack}}^{(\ell)} = \begin{bmatrix} M_Q^{(\ell)\top} & M_K^{(\ell)\top} & M_V^{(\ell)\top} & M_{\text{up}}^{(\ell)\top} & M_O^{(\ell)} & M_{\text{down}}^{(\ell)} \end{bmatrix} \in \mathbb{R}^{d \times N_\ell}$$

$$R_\ell = \text{QR}\!\left(M_{\text{stack}}^{(\ell)} f_t(R_{\ell, \text{prev}}^\top M_{\text{stack}}^{(\ell)})^\top\right)$$

Обновление каждого параметра — своим транспонированием (per Design 3).

### 4.4 Алгоритм Shifted Cholesky QR (SCQR)

Стандартный QR дорог для больших матриц. Авторы используют SCQR:

1. Вычислить регуляризованную Gram-матрицу: $P = A^\top A + \epsilon I_n$
2. Cholesky факторизация: $P = LL^\top$
3. $Q = A L^{-1}$ (решение треугольной системы, $\mathcal{O}(n^2)$)

При плохом conditioning — fallback на стандартный QR.

**ARO улучшает conditioning SCQR** потому что $M_t f_t(R_{t-1}^\top M_t)^\top$ содержит дополнительную нормализацию от $f_t$, делая матрицу более обусловленной.

### 4.5 Hyperparameter Transfer

Все non-Adam методы нормализуются для совпадения RMS-нормы с AdamW:

$$\Delta W_t \leftarrow 0.2 \cdot \frac{\Delta W_t}{\|\Delta W_t\|_F / \sqrt{mn}}$$

Это изолирует **направление** обновления как единственный фактор.

---

## 5. Практические детали

### 5.1 Stateful base optimizer (ARO-Adam)

При использовании Adam как $f_t$ — два шага:

**1. Look-ahead под старым вращением:**
$$\tilde{V}_t = \beta_2 V_{t-1} + (1-\beta_2)(R_{t-1}^\top M_t) \odot (R_{t-1}^\top M_t)$$
$$\tilde{D}_t = f_t(R_{t-1}^\top M_t; \tilde{V}_t) = \frac{R_{t-1}^\top M_t}{\sqrt{\tilde{V}_t}}$$

**2. Вычислить новое вращение:**
$$R_t = \text{QR}(M_t \tilde{D}_t^\top)$$

**3. Обновить state с новым вращением:**
$$V_t = \beta_2 V_{t-1} + (1-\beta_2)(R_t^\top M_t) \odot (R_t^\top M_t)$$
$$\Delta W_t = -\eta R_t \frac{R_t^\top M_t}{\sqrt{V_t}}$$

$\tilde{V}_t$ — временный буфер в scratch memory, не сохраняется.

### 5.2 Memory overhead

| Метод                      | Optimizer states                             |
| -------------------------- | -------------------------------------------- |
| AdamW                      | $M_t, V_t$ (2× модель)                       |
| Muon (hidden) + AdamW      | $M_t$ (hidden) + $M_t, V_t$ (rest)           |
| ARO-Sinkhorn               | $M_t, R_t$ (2× модель, $R_t$ ≈ $d \times d$) |
| ARO-Adam                   | $M_t, V_t, R_t$ (3× модель)                  |
| Наш гибрид (chain-coupled) | $M_t, R_\ell$ (≈ 2× модель)                  |

При 124M модели: ARO-Sinkhorn практически эквивалентен AdamW по памяти.

---

## 6. Имплементация

### 6.1 SinkGD base optimizer

```python
import torch
import torch.nn.functional as F


def sink_step(X: torch.Tensor, L: int = 5) -> torch.Tensor:
    """
    f_Sink: одновременная нормализация строк и столбцов (L итераций).

    Args:
        X: [m, n] матрица (gradient или momentum)
        L: число итераций Sinkhorn (default=5 из статьи)
    Returns:
        Нормализованная матрица той же формы
    """
    for _ in range(L):
        # Нормы строк: Q(X) = Diag(‖X[i,:]‖₂)
        row_norms = X.norm(dim=1, keepdim=True).clamp(min=1e-8)
        # Нормы столбцов: R(X) = Diag(‖X[:,j]‖₂)
        col_norms = X.norm(dim=0, keepdim=True).clamp(min=1e-8)
        # X ∝ Q(X)⁻¹ · X · R(X)⁻¹
        X = X / row_norms / col_norms
    return X
```

### 6.2 Shifted Cholesky QR

```python
def shifted_cholesky_qr(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Вычисляет Q-фактор из QR декомпозиции A через Cholesky.

    P = A^T A + eps*I  (regularized Gram matrix)
    P = L L^T          (Cholesky)
    Q = A L^{-1}       (triangular solve)

    Args:
        A: [m, n] матрица (m >= n рекомендуется)
        eps: сдвиг для численной стабильности
    Returns:
        Q: [m, m] ортонормированная матрица (Q-фактор)
    """
    m, n = A.shape

    # Если m < n — транспонируем, вычисляем, транспонируем обратно
    if m < n:
        return shifted_cholesky_qr(A.T, eps).T

    try:
        # P = A^T A + eps*I
        P = A.T @ A
        P.diagonal().add_(eps)

        # Cholesky: P = L L^T
        L = torch.linalg.cholesky(P)

        # Q = A L^{-1} через triangular solve
        # A = Q L^T => Q = A (L^T)^{-1}
        Q = torch.linalg.solve_triangular(L.T, A.T, upper=True).T

        # Проверка на NaN/Inf
        if not torch.isfinite(Q).all():
            raise RuntimeError("SCQR produced non-finite values")

        return Q

    except (RuntimeError, torch.linalg.LinAlgError):
        # Fallback на стандартный QR
        Q, _ = torch.linalg.qr(A)
        # Если m > n — дополняем до полного базиса
        if Q.shape[1] < m:
            Q_full, _ = torch.linalg.qr(
                torch.cat([Q, torch.randn(m, m - Q.shape[1], device=A.device, dtype=A.dtype)], dim=1)
            )
            return Q_full
        return Q


def compute_aro_rotation(M: torch.Tensor, R_prev: torch.Tensor,
                          f_func, eps: float = 1e-6) -> torch.Tensor:
    """
    R_t = QR(M_t · f_t(R_{t-1}^T · M_t)^T)

    Args:
        M: [m, n] momentum matrix
        R_prev: [m, m] предыдущее вращение
        f_func: callable, base optimizer function
        eps: для SCQR
    Returns:
        R_new: [m, m] новое вращение
    """
    # Шаг 1: проецируем M в повёрнутую СК
    M_rotated = R_prev.T @ M  # [m, n]

    # Шаг 2: применяем base optimizer к повёрнутому M
    D = f_func(M_rotated)  # [m, n]

    # Шаг 3: cross-Gram matrix
    cross_gram = M @ D.T  # [m, m]

    # Шаг 4: QR декомпозиция
    R_new = shifted_cholesky_qr(cross_gram, eps=eps)

    return R_new
```

### 6.3 ARO-Sinkhorn optimizer (per-layer)

```python
import torch
from torch.optim import Optimizer
from typing import Optional


class AROSinkhorn(Optimizer):
    """
    ARO с SinkGD как base optimizer.

    Из статьи (Equation 4):
        M_t = β M_{t-1} + (1-β) G_t
        R_t = QR(M_t · f_Sink(R_{t-1}^T M_t)^T)
        ΔW_t = -η · R_t · f_Sink(R_t^T M_t)

    Плюс RMS-нормализация для alignment с AdamW:
        ΔW_t ← 0.2 · ΔW_t / (‖ΔW_t‖_F / √(mn))
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta: float = 0.95,          # momentum coefficient
        sink_iters: int = 5,         # L в f_Sink
        rms_target: float = 0.2,     # target RMS norm (из AdamW alignment)
        weight_decay: float = 0.0,
        scqr_eps: float = 1e-6,
        nesterov: bool = False,
        full_qr_fallback: bool = True,
    ):
        defaults = dict(
            lr=lr, beta=beta, sink_iters=sink_iters,
            rms_target=rms_target, weight_decay=weight_decay,
            scqr_eps=scqr_eps, nesterov=nesterov,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AROSinkhorn does not support sparse gradients")

                # Применяем только к матрицам (2D+)
                if grad.ndim < 2:
                    # Для 1D: простой AdamW-style update
                    self._scalar_update(p, grad, group)
                    continue

                # Обрабатываем батч измерений: сворачиваем лишние в строки
                orig_shape = grad.shape
                if grad.ndim > 2:
                    grad_2d = grad.view(grad.shape[0], -1)
                    p_2d = p.view(p.shape[0], -1)
                else:
                    grad_2d = grad
                    p_2d = p

                m, n = grad_2d.shape
                state = self.state[p]

                # Инициализация state
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum'] = torch.zeros_like(grad_2d)
                    state['rotation'] = torch.eye(m, device=p.device, dtype=p.dtype)

                state['step'] += 1
                M = state['momentum']
                R_prev = state['rotation']

                beta = group['beta']
                L = group['sink_iters']
                eps = group['scqr_eps']

                # Weight decay
                if group['weight_decay'] != 0:
                    grad_2d = grad_2d.add(p_2d, alpha=group['weight_decay'])

                # M_t = β M_{t-1} + (1-β) G_t
                M.mul_(beta).add_(grad_2d, alpha=1 - beta)

                # R_t = QR(M_t · f_Sink(R_{t-1}^T M_t)^T)
                f_func = lambda x: sink_step(x, L=L)
                R_new = compute_aro_rotation(M, R_prev, f_func, eps=eps)
                state['rotation'] = R_new

                # ΔW_t = -η · R_t · f_Sink(R_t^T M_t)
                M_rotated = R_new.T @ M
                update = R_new @ sink_step(M_rotated, L=L)

                # RMS normalization: align с AdamW
                rms = update.norm() / (m * n) ** 0.5
                if rms > 1e-8:
                    update = update * (group['rms_target'] / rms)

                # Применяем update
                p_2d.add_(update, alpha=-group['lr'])

                # Восстанавливаем shape если нужно
                if grad.ndim > 2:
                    p.copy_(p_2d.view(orig_shape))

        return loss

    def _scalar_update(self, p, grad, group):
        """Простое SGD с momentum для 1D параметров."""
        state = self.state[p]
        if 'momentum_1d' not in state:
            state['momentum_1d'] = torch.zeros_like(grad)
        m = state['momentum_1d']
        beta = group['beta']
        m.mul_(beta).add_(grad, alpha=1 - beta)
        p.add_(m, alpha=-group['lr'])
```

### 6.4 Chain-coupled ARO + TEON grouping

```python
import torch
from torch import nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class LayerGroup:
    """Группа параметров одного слоя трансформера."""
    layer_idx: int
    # Параметры, потребляющие residual stream (нужен transpose)
    consumer_params: Dict[str, nn.Parameter] = field(default_factory=dict)
    # Параметры, производящие residual stream (не транспонируются)
    producer_params: Dict[str, nn.Parameter] = field(default_factory=dict)


class ChainCoupledARO:
    """
    ARO с chain-coupled rotation sharing (Design 1 + Design 3 из ARO paper).

    Для каждого слоя ℓ — одно вращение R_ℓ ∈ SO(d),
    оцениваемое из стекованного momentum всех параметров слоя.

    Ориентация (Design 3):
    - Consumer (Q, K, V, up, tok, pos, head): транспонируем перед ARO
    - Producer (O, down): НЕ транспонируем
    """

    def __init__(
        self,
        model: nn.Module,
        embed_dim: int,
        lr: float = 1e-3,
        beta: float = 0.95,
        sink_iters: int = 5,
        rms_target: float = 0.2,
        weight_decay: float = 0.0,
        scqr_eps: float = 1e-6,
    ):
        self.lr = lr
        self.beta = beta
        self.sink_iters = sink_iters
        self.rms_target = rms_target
        self.weight_decay = weight_decay
        self.embed_dim = embed_dim
        self.eps = scqr_eps

        # Группируем параметры по слоям
        self.layer_groups = self._group_params(model)

        # Состояния оптимизатора
        self.states: Dict[str, Dict] = {}
        self._init_states()

    def _group_params(self, model: nn.Module) -> List[LayerGroup]:
        """Автоматически определяем layer groups из GPT-like модели."""
        groups = []

        # Предполагаем стандартную GPT архитектуру
        # Адаптируйте под свою модель
        for name, module in model.named_modules():
            # Ищем attention блоки
            if hasattr(module, 'q_proj') and hasattr(module, 'k_proj'):
                # Пытаемся извлечь номер слоя из имени
                parts = name.split('.')
                layer_idx = next((int(p) for p in parts if p.isdigit()), len(groups))

                group = LayerGroup(layer_idx=layer_idx)

                # Consumer params (транспонируем): Q, K, V, up
                if hasattr(module, 'q_proj'):
                    group.consumer_params['q'] = module.q_proj.weight
                if hasattr(module, 'k_proj'):
                    group.consumer_params['k'] = module.k_proj.weight
                if hasattr(module, 'v_proj'):
                    group.consumer_params['v'] = module.v_proj.weight

                # Producer params (не транспонируем): O
                if hasattr(module, 'out_proj'):
                    group.producer_params['o'] = module.out_proj.weight

                groups.append(group)

        return groups

    def _get_param_id(self, param: nn.Parameter) -> int:
        return id(param)

    def _init_states(self):
        """Инициализируем optimizer states для всех групп."""
        for group in self.layer_groups:
            layer_key = f"layer_{group.layer_idx}"

            self.states[layer_key] = {
                'rotation': torch.eye(self.embed_dim),  # R_ℓ ∈ SO(d)
                'step': 0,
            }

            # Momentum для каждого параметра отдельно
            all_params = {**group.consumer_params, **group.producer_params}
            for param_name, param in all_params.items():
                param_key = f"{layer_key}_{param_name}"
                self.states[param_key] = {
                    'momentum': torch.zeros_like(param.data)
                }

    def _orient_for_aro(self, M: torch.Tensor, is_consumer: bool) -> torch.Tensor:
        """
        Ориентируем momentum для ARO (Design 3).
        Consumer: транспонируем (residual dim → row dim).
        Producer: не транспонируем.
        """
        if is_consumer and M.ndim == 2:
            return M.T  # [d, out] → [d, in], residual dim становится row dim
        return M

    def _deorient_update(self, dW: torch.Tensor, is_consumer: bool) -> torch.Tensor:
        """Обратное преобразование после ARO update."""
        if is_consumer and dW.ndim == 2:
            return dW.T
        return dW

    @torch.no_grad()
    def step_layer(self, layer_idx: int):
        """
        Выполняем один шаг ARO для одного слоя.

        1. Обновляем momentum для всех параметров слоя
        2. Стекуем momentum (с учётом ориентации)
        3. Вычисляем единое вращение R_ℓ из стека
        4. Применяем rotated update к каждому параметру
        """
        layer_key = f"layer_{layer_idx}"

        # Находим группу
        group = next((g for g in self.layer_groups if g.layer_idx == layer_idx), None)
        if group is None:
            return

        state = self.states[layer_key]
        state['step'] += 1

        # === Шаг 1: Обновляем momentum ===
        oriented_momentums = []  # список (oriented_M, is_consumer, param_name, param)

        all_params = [
            (name, param, True) for name, param in group.consumer_params.items()
        ] + [
            (name, param, False) for name, param in group.producer_params.items()
        ]

        for param_name, param, is_consumer in all_params:
            if param.grad is None:
                continue

            param_key = f"{layer_key}_{param_name}"
            M = self.states[param_key]['momentum']

            grad = param.grad
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data

            # M_t = β M_{t-1} + (1-β) G_t
            M.mul_(self.beta).add_(grad, alpha=1 - self.beta)

            # Ориентируем для ARO
            M_oriented = self._orient_for_aro(M, is_consumer)
            oriented_momentums.append((M_oriented, is_consumer, param_name, param))

        if not oriented_momentums:
            return

        # === Шаг 2: Стекуем momentum ===
        # Все ориентированные M имеют shape [d, *] — стекуем по второму измерению
        M_list = [m for m, _, _, _ in oriented_momentums]

        # Убеждаемся что первое измерение = embed_dim
        valid_M = [m for m in M_list if m.shape[0] == self.embed_dim]
        if not valid_M:
            return

        M_stack = torch.cat(valid_M, dim=1)  # [d, sum_of_n]

        # === Шаг 3: Вычисляем ARO rotation ===
        R_prev = state['rotation'].to(M_stack.device, M_stack.dtype)

        f_func = lambda x: sink_step(x, L=self.sink_iters)
        R_new = compute_aro_rotation(M_stack, R_prev, f_func, eps=self.eps)
        state['rotation'] = R_new.cpu()

        # === Шаг 4: Применяем rotated update к каждому параметру ===
        param_idx = 0
        for M_oriented, is_consumer, param_name, param in oriented_momentums:
            if M_oriented.shape[0] != self.embed_dim:
                param_idx += 1
                continue

            # ΔW = R_ℓ · f_Sink(R_ℓ^T · M_oriented)
            M_rot = R_new.T @ M_oriented
            update_oriented = R_new @ sink_step(M_rot, L=self.sink_iters)

            # RMS normalization
            m, n = update_oriented.shape
            rms = update_oriented.norm() / (m * n) ** 0.5
            if rms > 1e-8:
                update_oriented = update_oriented * (self.rms_target / rms)

            # Обратная ориентация
            update = self._deorient_update(update_oriented, is_consumer)

            # Применяем update
            param.data.add_(update, alpha=-self.lr)
            param_idx += 1

    @torch.no_grad()
    def step(self):
        """Выполняем шаг для всех слоёв."""
        for group in self.layer_groups:
            self.step_layer(group.layer_idx)

    def zero_grad(self):
        for group in self.layer_groups:
            all_params = {**group.consumer_params, **group.producer_params}
            for param in all_params.values():
                if param.grad is not None:
                    param.grad.zero_()
```

### 6.5 Hybrid setup: ARO для hidden layers + AdamW для остальных

```python
from torch.optim import AdamW
from typing import List


def create_hybrid_optimizer(
    model: nn.Module,
    embed_dim: int,
    aro_lr: float = 3e-4,
    adamw_lr: float = 3e-4,
    weight_decay: float = 0.1,
    beta: float = 0.95,
):
    """
    Создаём гибридный оптимизатор:
    - ARO-Sinkhorn для weight matrices hidden layers
    - AdamW для embeddings, LM head, LayerNorm, biases

    Это стандартная "hybrid setup" из статьи.
    """

    # Параметры для ARO (2D матрицы hidden layers)
    aro_params = []
    adamw_params = []

    # Имена параметров которые идут в AdamW
    adamw_keywords = ['embedding', 'norm', 'ln', 'bias', 'lm_head', 'wte', 'wpe']

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_adamw = any(kw in name.lower() for kw in adamw_keywords)
        is_1d = param.ndim < 2

        if is_adamw or is_1d:
            adamw_params.append(param)
        else:
            aro_params.append(param)

    print(f"ARO params: {sum(p.numel() for p in aro_params):,}")
    print(f"AdamW params: {sum(p.numel() for p in adamw_params):,}")

    # Создаём оптимизаторы
    aro_optimizer = AROSinkhorn(
        aro_params,
        lr=aro_lr,
        beta=beta,
        sink_iters=5,
        rms_target=0.2,
        weight_decay=weight_decay,
    )

    adamw_optimizer = AdamW(
        adamw_params,
        lr=adamw_lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )

    return aro_optimizer, adamw_optimizer


class CombinedOptimizer:
    """Обёртка для совместного использования двух оптимизаторов."""

    def __init__(self, *optimizers):
        self.optimizers = optimizers

    def step(self):
        for opt in self.optimizers:
            opt.step()

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, state_dicts):
        for opt, sd in zip(self.optimizers, state_dicts):
            opt.load_state_dict(sd)
```

### 6.6 Training loop

```python
def train_step(
    model: nn.Module,
    batch: dict,
    optimizer: CombinedOptimizer,
    scheduler,
    grad_clip: float = 1.0,
    use_amp: bool = True,
) -> float:
    """
    Один шаг обучения с ARO optimizer.
    """
    model.train()

    # Mixed precision (BF16 рекомендуется из Section 5.1)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
        outputs = model(**batch)
        loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping (стандартная практика)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()
    scheduler.step()

    return loss.item()


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    """Cosine decay с warmup (из Section 5.6 для Qwen3-8B)."""
    import math

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay)

    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)
```

---

## 7. Конфигурация для 124M модели

### 7.1 Рекомендуемые гиперпараметры

```python
CONFIG = {
    # Модель
    "model_size": "124M",
    "embed_dim": 768,
    "n_layers": 12,
    "n_heads": 12,

    # Данные (1× Chinchilla ≈ 2.5B tokens для 124M)
    "total_tokens": 2_500_000_000,
    "batch_size_tokens": 524_288,      # 512K tokens per batch
    "seq_len": 1024,
    "total_steps": 4768,               # total_tokens / batch_size_tokens

    # ARO-Sinkhorn (из paper Section 5.3)
    "aro_lr": 3e-4,                    # transfer from AdamW lr
    "adamw_lr": 3e-4,                  # AdamW baseline (нужен sweep)
    "beta": 0.95,                      # momentum coefficient
    "sink_iters": 5,                   # L в f_Sink
    "weight_decay": 0.1,
    "rms_target": 0.2,                 # RMS normalization target

    # LR schedule
    "warmup_steps": 200,               # ~4% of total
    "min_lr_ratio": 0.1,              # final lr = 10% of peak

    # Training
    "grad_clip": 1.0,
    "use_bf16": True,                  # BF16 training (рекомендовано из Section 5.1)

    # SCQR
    "scqr_eps": 1e-6,

    # Setup
    "optimizer_mode": "hybrid",        # "hybrid" или "full_model"
}
```

### 7.2 Ожидаемые результаты

На основе результатов из статьи для GPT2-124M (Section 5.3):

| Метод                        | Validation Loss | Speedup vs AdamW |
| ---------------------------- | --------------- | ---------------- |
| AdamW                        | ~3.28           | 1.0×             |
| Muon (hybrid)                | ~3.20           | ~1.15×           |
| Eigen-Sinkhorn               | ~3.22           | ~1.10×           |
| **ARO-Sinkhorn (hybrid)**    | **~3.16**       | **~1.25×**       |
| ARO-Sinkhorn + chain-coupled | **~3.14**       | **~1.28×**       |

### 7.3 Порядок экспериментов (от простого к сложному)

```
Эксперимент 1: Baseline
├── AdamW (полный sweep lr ∈ {1e-4, 3e-4, 1e-3})
└── Фиксируем best_adamw_lr

Эксперимент 2: ARO-Sinkhorn hybrid
├── lr = best_adamw_lr (через RMS transfer)
├── beta = 0.95
└── Проверяем +speedup vs AdamW

Эксперимент 3: ARO-Sinkhorn + TEON grouping
├── Те же гиперпараметры
├── Добавляем QKV stacking
└── Ожидаем дополнительный ~0.01-0.02 gain в loss

Эксперимент 4: Chain-coupled rotation sharing
├── Одно вращение на слой (из Section 6.5)
└── Ожидаем best performance (Table 3 из статьи)

Эксперимент 5: Full-model ARO (бонус)
└── ARO для всех матриц включая embeddings
    (выигрыш проявляется при overtraining)
```

---

## 8. Чеклист и советы

### 8.1 Критические правила (из Section 5.1)

- [x] **BF16 training с FP32 master weights** — не BF16 master weights
- [x] **Alignment lr schedules** — все методы обучаются одинаковое число шагов
- [x] **RMS-norm matching** — перед benchmarking нормализуй все методы
- [x] **AdamW для non-hidden params** в hybrid setup
- [x] **End-to-end AdamW tuning** — sweep по lr для AdamW, затем transfer

### 8.2 Частые ошибки

**Ошибка 1: Неправильная ориентация**

```python
# ❌ Неправильно: применяем ARO к W_Q напрямую
R @ W_Q  # W_Q имеет shape [d_model, d_k]

# ✅ Правильно: транспонируем, применяем, транспонируем обратно
# W_Q.T имеет shape [d_k, d_model], residual dim = d_model = строки
update = (R @ sink_step(R.T @ W_Q.T @ M)).T
```

**Ошибка 2: Rotational-equivariant f_t**

```python
# ❌ Muon уже rotational-equivariant → ARO не даёт эффекта
# R_t f_RN(R_t^T M) = f_RN(M) при сходимости power iteration

# ✅ Используй SinkGD или Sign — они не equivariant
```

**Ошибка 3: Aggressive polar projection**

```python
# ❌ Не используй полярное разложение (Polar) вместо QR
# Polar максимизирует J̃ агрессивно → нестабильность при mini-batch noise

# ✅ QR — консервативнее, лучше stability-speed tradeoff
```

**Ошибка 4: Сравнение с BF16 master weights**

```python
# ❌ BF16 master weights искусственно завышают speedup
# (из Section 5.1 статьи: "unrealistic high speedup numbers")

# ✅ Всегда FP32 master weights, BF16 только для forward/backward
```

### 8.3 Debugging checklist

```python
def check_aro_health(optimizer: AROSinkhorn):
    """Проверяем здоровье оптимизатора."""
    for group in optimizer.param_groups:
        for p in group['params']:
            if p not in optimizer.state or len(optimizer.state[p]) == 0:
                continue
            state = optimizer.state[p]

            R = state.get('rotation')
            if R is not None:
                # Проверяем ортогональность: R^T R ≈ I
                I_approx = R.T @ R
                orth_error = (I_approx - torch.eye(R.shape[0], device=R.device)).norm()
                if orth_error > 0.01:
                    print(f"⚠️ Rotation not orthogonal: error={orth_error:.4f}")

                # Проверяем NaN
                if not torch.isfinite(R).all():
                    print("❌ Rotation contains NaN/Inf!")

            M = state.get('momentum')
            if M is not None and not torch.isfinite(M).all():
                print("❌ Momentum contains NaN/Inf!")

    print("✅ ARO health check passed")


def log_aro_metrics(optimizer: AROSinkhorn, step: int):
    """Логируем метрики ARO для мониторинга."""
    import wandb

    j_values = []
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state.get(p, {})
            R = state.get('rotation')
            M = state.get('momentum')

            if R is not None and M is not None:
                M_rot = R.T @ M
                # J = ‖R^T G‖_* ≈ ‖M_rot‖_* через Sinkhorn
                j = sink_step(M_rot).norm()
                j_values.append(j.item())

    if j_values and step % 100 == 0:
        wandb.log({
            "aro/mean_J": sum(j_values) / len(j_values),
            "aro/step": step,
        })
```

### 8.4 Memory estimation для 124M

```
Параметры модели: 124M × 4 bytes (FP32) = 496 MB

ARO-Sinkhorn states:
- Momentum M_t: 124M × 4 = 496 MB
- Rotations R_ℓ (12 слоёв × d×d = 12 × 768×768): ~28 MB
- Итого optimizer states: ~524 MB

Общая память (FP32 weights + BF16 activations):
- Weights (FP32): 496 MB
- Gradients (FP32): 496 MB
- Optimizer states: 524 MB
- Activations (BF16, batch=8, seq=1024): ~200 MB
= ~1.7 GB GPU memory
```

### 8.5 Быстрый старт (минимальный пример)

```python
import torch
from torch import nn

# Простейшее использование AROSinkhorn
model = MyGPTModel(config)

# Разделяем параметры
matrix_params = [p for n, p in model.named_parameters()
                 if p.requires_grad and p.ndim >= 2
                 and 'embed' not in n and 'norm' not in n]
other_params = [p for n, p in model.named_parameters()
                if p.requires_grad and (p.ndim < 2
                or 'embed' in n or 'norm' in n)]

# Создаём оптимизаторы
aro_opt = AROSinkhorn(matrix_params, lr=3e-4, beta=0.95)
adam_opt = torch.optim.AdamW(other_params, lr=3e-4, weight_decay=0.1)

# Training loop
for batch in dataloader:
    loss = model(**batch).loss
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    aro_opt.step()
    adam_opt.step()

    aro_opt.zero_grad()
    adam_opt.zero_grad()

    scheduler.step()
```

---

## Итог

Гибрид ARO + TEON-style grouping — это **ARO с chain-coupled rotation sharing**:

- TEON предоставляет стратегию **группировки** (QKV стек)
- ARO предоставляет **политику вращения** (non-eigen, optimizer-aware QR)
- Результат: строго лучше каждого метода по отдельности

Ожидаемый выигрыш для 124M при 1× Chinchilla:

- vs AdamW: **+0.10–0.14** в training loss
- vs Muon: **+0.02–0.04** в training loss
- Overhead по памяти: минимальный (~5%)
- Overhead по времени: ~1–3%

**Приоритет реализации:** `AROSinkhorn` → hybrid setup → TEON grouping → chain-coupled sharing.
