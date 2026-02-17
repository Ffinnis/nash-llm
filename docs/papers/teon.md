# TEON: Tensorized Orthonormalization Beyond Layer-Wise MUON

## Краткая суть

TEON — обобщение MUON-оптимизатора, которое вместо независимой ортогонализации градиентов **каждого слоя по отдельности** стакает градиенты **одинаковых слоёв из соседних блоков** в тензор и ортогонализирует его целиком. Это позволяет учитывать корреляции между слоями.

---

## Математика

### MUON (baseline)

Для слоя $W_t \in \mathbb{R}^{m \times n}$ с градиентом $G_t$:

$$M_t = \mu M_{t-1} + (1 - \mu) G_t$$

$$O_t = \text{Ortho}(M_t) \quad \text{где} \quad \text{Ortho}(M) = UV^T \quad \text{из SVD } M = U\Sigma V^T$$

$$W_t = W_{t-1} - \eta \sqrt{m/n} \cdot O_t$$

### TEON: ключевые операции

**1. Стакинг градиентов в тензор**

Берём $K$ одинаковых слоёв (например, Q-проекции из $K$ последовательных transformer-блоков) и стакаем их моментумы:

$$\mathcal{T} = \text{Ten}(M^{(1)}, \ldots, M^{(K)}) \in \mathbb{R}^{m \times n \times K}$$

где $\mathcal{T}[:, :, k] = M^{(k)}$.

**2. Mode-1 матрицизация**

Разворачиваем тензор вдоль первой моды:

$$\mathcal{M}_1(\mathcal{T}) \in \mathbb{R}^{m \times (nK)}$$

Конкретно: конкатенация слайсов по второй оси:

$$\mathcal{M}_1(\mathcal{T}) = [M^{(1)} | M^{(2)} | \ldots | M^{(K)}]$$

**3. Ортогонализация**

$$Z = \mathcal{M}_1(\mathcal{T})$$

$$Q = \text{Ortho}(Z) = UV^T \quad \text{из SVD } Z = U\Sigma V^T$$

**4. Обратная свёртка и обновление**

$$\mathcal{O} = \mathcal{M}_1^{-1}(Q) \in \mathbb{R}^{m \times n \times K}$$

$$W^{(k)}_t = W^{(k)}_{t-1} - \eta \sqrt{m/n} \cdot \mathcal{O}[:, :, k]$$

### Приближение SVD: методы ортогонализации

Вместо точного SVD ($O(\min(mn^2, nm^2))$) используются итеративные полиномиальные методы, которые работают только с матричными умножениями (GPU-friendly).

Все методы применяют нечётный полином $p(x) = ax + bx^3 + cx^5$ к сингулярным значениям матрицы. Если $M = U\Sigma V^T$, то $p(M) = Up(\Sigma)V^T$, и при сходимости $p(\sigma_i) \to 1$ для всех $\sigma_i$, что даёт $p(M) \to UV^T = \text{polar}(M)$.

Каждая итерация вычисляется через матричные умножения:

$$X_{t} = a \cdot X_{t-1} + Y_{t-1}(b \cdot I + c \cdot Y_{t-1}) \cdot X_{t-1}, \quad \text{где } Y_{t-1} = X_{t-1}^T X_{t-1}$$

---

## Polar Express: оптимальная ортогонализация (подробно)

### Суть метода

**Polar Express** (Amsel et al., 2025) — адаптивный полиномиальный метод вычисления полярного разложения $\text{polar}(M) = UV^T$. В отличие от Newton-Schulz и Jordan, которые используют **одинаковый** полином на каждой итерации, Polar Express **адаптирует коэффициенты** от итерации к итерации, решая задачу минимаксной аппроксимации функции $\text{sign}(x)$ на текущем интервале сингулярных значений.

### Почему это лучше

| Метод               | Сходимость                                | Точность за 5 итераций | Проблемы                                         |
| ------------------- | ----------------------------------------- | ---------------------- | ------------------------------------------------ |
| Newton-Schulz (d=5) | Сверхэкспоненциальная, но медленный старт | Низкая                 | Медленная начальная фаза                         |
| Jordan              | Быстрый старт                             | Средняя (~0.3 ошибка)  | **Не сходится** — плато                          |
| You (6 шагов)       | Быстрый старт                             | Средняя                | **Не сходится**, определён только для 6 итераций |
| **Polar Express**   | Сверхэкспоненциальная, быстрый старт      | **Наилучшая**          | —                                                |

### Математическая основа

Задача: найти композицию полиномов $p^* = p_T \circ p_{T-1} \circ \cdots \circ p_1$, минимизирующую worst-case ошибку:

$$p^* = \arg\min_{\substack{p = p_T \circ \cdots \circ p_1 \\ p_t \in \mathcal{P}^{\text{odd}}_d}} \max_{x \in [\ell, u]} |1 - p(x)|$$

**Ключевая теорема (Theorem 4.1):** жадный выбор полинома на каждой итерации даёт **глобально оптимальную** композицию.

Алгоритм:

1. Инициализация: $\ell_1 = \ell$ (нижняя граница сингулярных значений), $u_1 = u = 1$
2. На шаге $t$: решаем минимаксную задачу

$$p_t = \arg\min_{p \in \mathcal{P}^{\text{odd}}_d} \max_{x \in [\ell_t, u_t]} |1 - p(x)|$$

3. Обновляем границы: $\ell_{t+1} = p_t(\ell_t)$, $u_{t+1} = 2 - \ell_{t+1}$

Ошибка после $T$ итераций:

$$\|\text{polar}(M) - X_T\|_2 \leq |1 - \ell^2|^{(q+1)^T}$$

Для degree-5 ($q=2$) — **кубическая** сходимость.

### Решение минимаксной задачи для degree-5

Для $p(x) = ax + bx^3 + cx^5$ ищем коэффициенты через **теорему равноосцилляции** (Equioscillation Theorem): оптимальный полином должен достигать максимальной ошибки $\pm E$ в 4 точках $\{\ell, q, r, u\}$ с чередующимися знаками:

$$p(\ell) = 1 - E, \quad p(q) = 1 + E, \quad p(r) = 1 - E, \quad p(u) = 1 + E$$

Это линейная система $4 \times 4$:

$$\begin{pmatrix} \ell & \ell^3 & \ell^5 & 1 \\ q & q^3 & q^5 & -1 \\ r & r^3 & r^5 & 1 \\ u & u^3 & u^5 & -1 \end{pmatrix} \begin{pmatrix} a \\ b \\ c \\ E \end{pmatrix} = \begin{pmatrix} 1 \\ 1 \\ 1 \\ 1 \end{pmatrix}$$

Точки $q, r$ — внутренние экстремумы $p'(x) = 0$, находятся через квадратичную формулу:

$$q = \sqrt{\frac{-3b - \sqrt{9b^2 - 20ac}}{10c}}, \quad r = \sqrt{\frac{-3b + \sqrt{9b^2 - 20ac}}{10c}}$$

Итерируем: решаем систему → находим $q, r$ → решаем систему → ... (обычно 3-5 итераций до сходимости).

### Стабилизация для bfloat16

Три ключевых модификации:

**1. Safety factor (делим на 1.01):** Если из-за round-off $\sigma_i$ чуть больше $u_t$, полином может раздуть его до бесконечности. Замена $p_t(x) \to p_t(x / 1.01)$ стабилизирует (сингулярные значения сходятся к 0.999998, что в bfloat16 неотличимо от 1). На последней итерации safety factor **убирается**.

**2. Cushioning (подушка):** Оптимальный полином может отображать некоторые $\sigma_i$ близко к нулю или даже в отрицательные значения (non-monotonicity). Когда $\ell_t < u_t / 10$, используем $\ell_t = u_t / 10$ вместо истинного $\ell_t$. Это гарантирует $p_t(x)/x \geq 0.236$ для всех $x$.

**3. Нормализация с эпсилон:** $X_0 = M / (\|M\|_F + 10^{-2})$ вместо $M / \|M\|_F$ — защита от деления на ноль и стабилизация малых норм.

### Предвычисленные коэффициенты (offline stage)

Для $\ell = 10^{-3}$, $u = 1$, degree-5, с safety factor и cushioning:

```python
coeffs_list = [
    (8.28721201814563,    -23.595886519098837, 17.300387312530933),  # iter 1
    (4.107059111542203,   -2.9478499167379106, 0.5448431082926601),  # iter 2
    (3.9486908534822946,  -2.908902115962949,  0.5518191394370137),  # iter 3
    (3.3184196573706015,  -2.488488024314874,  0.51004894012372),    # iter 4
    (2.300652019954817,   -1.6689039845747493, 0.4188073119525673),  # iter 5
    (1.891301407787398,   -1.2679958271945868, 0.37680408948524835), # iter 6
    (1.8750014808534479,  -1.2500016453999487, 0.3750001645474248),  # iter 7
    (1.875,               -1.25,               0.375),               # iter 8+ (= Newton-Schulz)
]

# Safety factor для всех кроме последнего полинома
coeffs_list = [
    (a / 1.01, b / 1.01**3, c / 1.01**5)
    for (a, b, c) in coeffs_list[:-1]
] + [coeffs_list[-1]]
```

Обрати внимание: первый полином ($a \approx 8.3$, $b \approx -23.6$) сильно отличается от остальных — он агрессивно «поднимает» маленькие сингулярные значения. К 7-8 итерации коэффициенты стабилизируются на $(1.875, -1.25, 0.375) = (15/8, -10/8, 3/8)$ — это стандартный **Newton-Schulz degree-5**.

### Полная реализация Polar Express (PyTorch)

```python
from itertools import repeat
import torch

POLAR_EXPRESS_COEFFS = [
    (8.28721201814563,    -23.595886519098837, 17.300387312530933),
    (4.107059111542203,   -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946,  -2.908902115962949,  0.5518191394370137),
    (3.3184196573706015,  -2.488488024314874,  0.51004894012372),
    (2.300652019954817,   -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398,   -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479,  -1.2500016453999487, 0.3750001645474248),
    (1.875,               -1.25,               0.375),
]

# Safety factor для численной стабильности (кроме последнего полинома)
POLAR_EXPRESS_COEFFS_SAFE = [
    (a / 1.01, b / 1.01**3, c / 1.01**5)
    for (a, b, c) in POLAR_EXPRESS_COEFFS[:-1]
] + [POLAR_EXPRESS_COEFFS[-1]]


@torch.compile
def polar_express(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Polar Express: оптимальная полиномиальная аппроксимация polar(G) = UV^T.

    Использует предвычисленные коэффициенты для ℓ=1e-3, u=1, degree=5.
    Работает в bfloat16 для скорости на GPU.

    Args:
        G: входная матрица (градиент или моментум), ≥ 2D
        steps: число итераций (рекомендуется 5-6 для deep learning)

    Returns:
        Приближение к polar(G) = UV^T
    """
    assert G.ndim >= 2
    X = G.bfloat16()

    # Транспонируем если tall matrix — уменьшает FLOPs
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT

    # Нормализация: X0 = M / (||M||_F * 1.01 + eps)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)

    # Берём нужное число коэффициентов, добавляя Newton-Schulz для steps > 8
    hs = POLAR_EXPRESS_COEFFS_SAFE[:steps] + list(
        repeat(POLAR_EXPRESS_COEFFS_SAFE[-1], max(0, steps - len(POLAR_EXPRESS_COEFFS_SAFE)))
    )

    for a, b, c in hs:
        A = X @ X.mT                   # A = X X^T        (m × m)
        B = b * A + c * A @ A           # B = bA + cA^2    (m × m)
        X = a * X + B @ X              # X = aX + bX^3 + cX^5

    if transposed:
        X = X.mT

    return X
```

### Генерация коэффициентов (offline stage)

Если нужны коэффициенты для других $\ell$ или другого числа итераций:

```python
from math import inf, sqrt
import numpy as np


def optimal_quintic(l: float, u: float):
    """
    Находит оптимальный degree-5 нечётный полином p(x) = ax + bx^3 + cx^5,
    минимизирующий max_{x in [l,u]} |1 - p(x)| через алгоритм Ремеза.
    """
    assert 0 <= l <= u

    # Когда l ≈ u, оптимальный полином = scaled Newton-Schulz
    if 1 - 5e-6 <= l / u:
        return (15/8) / u, (-10/8) / (u**3), (3/8) / (u**5)

    # Инициализация trial points
    q = (3 * l + u) / 4
    r = (l + 3 * u) / 4
    E, old_E = inf, None

    # Remez-подобная итерация
    while not old_E or abs(old_E - E) > 1e-15:
        old_E = E
        LHS = np.array([
            [l, l**3, l**5, 1],
            [q, q**3, q**5, -1],
            [r, r**3, r**5, 1],
            [u, u**3, u**5, -1],
        ])
        a, b, c, E = np.linalg.solve(LHS, np.ones(4))
        discriminant = 9 * b**2 - 20 * a * c
        q = np.sqrt((-3 * b - sqrt(discriminant)) / (10 * c))
        r = np.sqrt((-3 * b + sqrt(discriminant)) / (10 * c))

    return float(a), float(b), float(c)


def compute_polar_express_coeffs(
    l: float = 1e-3,
    num_iters: int = 10,
    cushion: float = 0.02407327424182761,
):
    """
    Вычисляет коэффициенты Polar Express для заданных параметров.

    Args:
        l: нижняя граница сингулярных значений (ℓ)
        num_iters: число итераций
        cushion: порог для cushioning (ℓ_t = max(ℓ_t, cushion * u_t))

    Returns:
        Список кортежей (a, b, c) для каждой итерации
    """
    u = 1.0
    coefficients = []
    for _ in range(num_iters):
        a, b, c = optimal_quintic(max(l, cushion * u), u)

        # Перецентрируем полином вокруг 1 на [l, u]
        pl = a * l + b * l**3 + c * l**5
        pu = a * u + b * u**3 + c * u**5
        rescalar = 2 / (pl + pu)
        a *= rescalar
        b *= rescalar
        c *= rescalar

        coefficients.append((a, b, c))
        l = a * l + b * l**3 + c * l**5
        u = 2 - l

    return coefficients
```

### Результаты Polar Express на GPT-2

Из paper (FineWeb, 1B tokens):

| Модель           | AdamW | MUON-Jordan | MUON-You | MUON-PolarExpress |
| ---------------- | ----- | ----------- | -------- | ----------------- |
| GPT-Small (124M) | 4.197 | 3.639       | 3.629    | **3.588**         |
| GPT-Large (774M) | 4.172 | 3.398       | 3.400    | **3.340**         |

PolarExpress стабильно лучше на **всех** learning rates.

### Теоретический выигрыш

Граница сходимости TEON vs MUON:

$$\|\nabla f(W_{\tau_{\text{TEON}}})\|_* \leq \sqrt{\frac{2 L_{\text{TEON}} \Delta_0}{T}}$$

$$\|\nabla f(W_{\tau_{\text{MUON}}})\|_* \leq \sqrt{K} \cdot \sqrt{\frac{2 L_{\text{TEON}} \Delta_0}{T}}$$

В лучшем случае TEON быстрее MUON в $\sqrt{K}$ раз. Максимальный выигрыш достигается когда **ведущие правые сингулярные вектора** стакаемых слоёв выровнены (для mode-1).

---

## Практические рекомендации из paper

| Параметр                  | Рекомендация                                        | Причина                                                     |
| ------------------------- | --------------------------------------------------- | ----------------------------------------------------------- |
| Matricization mode        | **Mode-1**                                          | Правые сингулярные вектора Q/K/V лучше выровнены            |
| K (число стакаемых слоёв) | **K=2**                                             | При K>2 выравненность падает, approximate SVD деградирует   |
| Какие слои стакать        | **Только Q, K, V** (одного типа из соседних блоков) | MLP-градиенты не выровнены, создают ill-conditioned матрицы |
| SVD approximation         | **PolarExpress** (5 итераций)                       | Ближе всего к exact SVD                                     |

---

## Реализация на Python (PyTorch)

```python
import torch
from torch.optim import Optimizer
from typing import List, Optional
from itertools import repeat


# ============================================================
# Polar Express: предвычисленные коэффициенты (ℓ=1e-3, d=5)
# ============================================================

POLAR_EXPRESS_COEFFS = [
    (8.28721201814563,    -23.595886519098837, 17.300387312530933),
    (4.107059111542203,   -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946,  -2.908902115962949,  0.5518191394370137),
    (3.3184196573706015,  -2.488488024314874,  0.51004894012372),
    (2.300652019954817,   -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398,   -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479,  -1.2500016453999487, 0.3750001645474248),
    (1.875,               -1.25,               0.375),
]
# Safety factor (кроме последнего — тот уже = Newton-Schulz)
POLAR_EXPRESS_COEFFS_SAFE = [
    (a / 1.01, b / 1.01**3, c / 1.01**5)
    for (a, b, c) in POLAR_EXPRESS_COEFFS[:-1]
] + [POLAR_EXPRESS_COEFFS[-1]]


@torch.compile
def polar_express(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Polar Express: оптимальная полиномиальная аппроксимация polar(G).
    Из paper: Amsel et al., "The Polar Express", 2025.
    Работает в bfloat16 для максимальной скорости на GPU.
    """
    assert G.ndim >= 2
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    hs = POLAR_EXPRESS_COEFFS_SAFE[:steps] + list(
        repeat(POLAR_EXPRESS_COEFFS_SAFE[-1], max(0, steps - len(POLAR_EXPRESS_COEFFS_SAFE)))
    )
    for a, b, c in hs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


def orthogonalize(
    M: torch.Tensor,
    steps: int = 5,
    method: str = "polar_express",
) -> torch.Tensor:
    """
    Приближённая ортогонализация: M → UV^T.

    Methods:
        - "polar_express": адаптивные оптимальные полиномы (рекомендуется)
        - "jordan": фиксированные коэфф. из MUON (3.4445, -2.6056, 0.6775)
        - "newton_schulz": стандартный NS degree-5 (15/8, -10/8, 3/8)
    """
    if method == "polar_express":
        return polar_express(M, steps)

    transposed = False
    if M.shape[0] > M.shape[1]:
        M = M.T
        transposed = True

    X = M.bfloat16()
    X = X / (X.norm() * 1.01 + 1e-7)

    if method == "jordan":
        a, b, c = 3.4445, -2.6056, 0.6775
    elif method == "newton_schulz":
        a, b, c = 15/8, -10/8, 3/8
    else:
        raise ValueError(f"Unknown method: {method}")

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transposed:
        X = X.T
    return X


class TEON(Optimizer):
    """
    TEON optimizer: Tensorized Orthonormalization Beyond Layer-Wise MUON.

    Основная идея:
    1. Собираем моментумы одинаковых слоёв из K соседних transformer-блоков
    2. Конкатенируем их в mode-1 matricization: [M1 | M2 | ... | MK] ∈ R^{m x (n*K)}
    3. Ортогонализируем полученную матрицу
    4. Разрезаем обратно и обновляем каждый слой

    Args:
        teon_params: список списков параметров для TEON-групп.
                     Каждая группа — список из K параметров одного типа
                     из последовательных блоков, например:
                     [[block0.q, block1.q], [block0.k, block1.k], ...]
        lr: learning rate
        momentum: momentum coefficient (µ)
        weight_decay: weight decay
        ns_steps: число итераций ортогонализации
        ns_method: метод ортогонализации ("polar_express", "jordan", "newton_schulz")
    """

    def __init__(
        self,
        teon_params: List[List[torch.nn.Parameter]],
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        ns_method: str = "polar_express",
    ):
        # Flatten для регистрации в Optimizer
        all_params = []
        for group in teon_params:
            all_params.extend(group)

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            ns_method=ns_method,
        )
        super().__init__(all_params, defaults)

        # Сохраняем группировку: список групп, каждая — список из K параметров
        self.teon_groups = teon_params

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        defaults = self.defaults
        lr = defaults["lr"]
        mu = defaults["momentum"]
        wd = defaults["weight_decay"]
        ns_steps = defaults["ns_steps"]
        ns_method = defaults["ns_method"]

        for group in self.teon_groups:
            K = len(group)
            if K == 0:
                continue

            # Собираем градиенты и обновляем моментумы
            momentums = []
            for param in group:
                if param.grad is None:
                    continue

                grad = param.grad
                state = self.state[param]

                # Инициализация состояния
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(param.data)

                buf = state["momentum_buffer"]
                # M_t = µ * M_{t-1} + (1 - µ) * G_t
                buf.mul_(mu).add_(grad, alpha=1.0 - mu)
                momentums.append(buf)

            if len(momentums) != K:
                # Если не все параметры имеют градиенты — fallback на MUON per-layer
                for i, param in enumerate(group):
                    if param.grad is None:
                        continue
                    buf = self.state[param]["momentum_buffer"]
                    m, n = buf.shape
                    ortho = orthogonalize(buf, ns_steps, ns_method)
                    if wd > 0:
                        param.data.mul_(1.0 - lr * wd)
                    param.data.add_(ortho, alpha=-lr * (m / n) ** 0.5)
                continue

            m, n = momentums[0].shape

            # === TEON core: Mode-1 Matricization ===
            # Конкатенация: Z = [M^(1) | M^(2) | ... | M^(K)] ∈ R^{m x (n*K)}
            Z = torch.cat(momentums, dim=1)  # shape: (m, n*K)

            # Ортогонализация
            Q = orthogonalize(Z, ns_steps, ns_method)

            # Обратная свёртка: разрезаем Q обратно на K частей
            ortho_slices = Q.split(n, dim=1)  # K тензоров по (m, n)

            # Обновление параметров
            scale = (m / n) ** 0.5
            for i, param in enumerate(group):
                if wd > 0:
                    param.data.mul_(1.0 - lr * wd)
                param.data.add_(ortho_slices[i], alpha=-lr * scale)

        return loss


# ============================================================
# Вспомогательная функция для создания TEON-групп из модели
# ============================================================

def create_teon_groups(
    model: torch.nn.Module,
    K: int = 2,
    layer_types: List[str] = None,
) -> List[List[torch.nn.Parameter]]:
    """
    Создаёт группы параметров для TEON из transformer-модели.

    Предполагается стандартная структура:
      model.transformer.h[i].attn.{q_proj, k_proj, v_proj}
    или аналогичная.

    Args:
        model: transformer-модель
        K: число слоёв для стакинга (рекомендуется 2)
        layer_types: названия параметров для стакинга,
                     например ["q_proj.weight", "k_proj.weight", "v_proj.weight"]

    Returns:
        Список групп, каждая — список из K параметров
    """
    if layer_types is None:
        layer_types = ["q_proj.weight", "k_proj.weight", "v_proj.weight"]

    # Собираем параметры по типам
    # Структура: {layer_type: [param_block_0, param_block_1, ...]}
    param_by_type = {lt: [] for lt in layer_types}

    for name, param in model.named_parameters():
        for lt in layer_types:
            if lt in name:
                param_by_type[lt].append(param)
                break

    # Формируем группы по K последовательных блоков
    groups = []
    for lt in layer_types:
        params = param_by_type[lt]
        num_groups = len(params) // K
        for i in range(num_groups):
            group = params[i * K : (i + 1) * K]
            groups.append(group)

    return groups


# ============================================================
# Пример интеграции в training loop
# ============================================================

def example_training_setup(model):
    """
    Пример настройки TEON + AdamW для обучения transformer-модели.
    TEON применяется к Q/K/V, AdamW — к остальным параметрам.
    """
    # 1. Определяем TEON-группы (Q, K, V из соседних блоков, K=2)
    teon_groups = create_teon_groups(model, K=2)
    teon_param_set = set()
    for group in teon_groups:
        for p in group:
            teon_param_set.add(id(p))

    # 2. Остальные параметры — для AdamW
    adamw_params = [
        p for p in model.parameters()
        if id(p) not in teon_param_set and p.requires_grad
    ]

    # 3. Создаём оптимизаторы
    teon_optimizer = TEON(
        teon_params=teon_groups,
        lr=0.02,
        momentum=0.95,
        weight_decay=0.1,
        ns_steps=5,
        ns_method="polar_express",
    )

    adamw_optimizer = torch.optim.AdamW(
        adamw_params,
        lr=4e-4,
        weight_decay=0.1,
    )

    return teon_optimizer, adamw_optimizer


def train_step(model, batch, teon_opt, adamw_opt):
    """Один шаг обучения."""
    model.train()
    teon_opt.zero_grad()
    adamw_opt.zero_grad()

    loss = model(**batch).loss
    loss.backward()

    teon_opt.step()
    adamw_opt.step()

    return loss.item()
```

---

## Ключевые моменты для реализации

### Что стакать

```
Block i:     Q_i, K_i, V_i, O_i, MLP1_i, MLP2_i
Block i+1:   Q_{i+1}, K_{i+1}, V_{i+1}, ...

TEON-группы (K=2):
  [Q_i, Q_{i+1}]     — стак Q из блоков i и i+1
  [K_i, K_{i+1}]     — стак K из блоков i и i+1
  [V_i, V_{i+1}]     — стак V из блоков i и i+1
```

### Что НЕ стакать (оставить на AdamW)

- Embedding / Unembedding слои
- LayerNorm / RMSNorm
- Positional encodings
- O-проекции и MLP-слои (их можно обучать обычным MUON или AdamW)

### Mode-1 matricization в коде

Для тензора $\mathcal{T} \in \mathbb{R}^{m \times n \times K}$, mode-1 matricization — это просто `torch.cat` по dim=1:

```python
# T[:,:,0] = M1 ∈ R^{m×n}
# T[:,:,1] = M2 ∈ R^{m×n}
# Mode-1: [M1 | M2] ∈ R^{m × 2n}
Z = torch.cat([M1, M2], dim=1)
```

Обратная операция — `split`:

```python
O1, O2 = Z.split(n, dim=1)
```

### Dimensional scaling factor

$$\text{scale} = \sqrt{m/n}$$

Применяется при обновлении весов (из Bernstein, 2025), обеспечивает корректное масштабирование для прямоугольных матриц.
