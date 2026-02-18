# Spectra-TEON: Гибридный оптимизатор

Гибрид между **TEON** (кросс-слойная ортогонализация) и **Spectra** (spike-aware спектральное формирование). Вместо полного выравнивания спектра — локальное подавление spike на объединённой тензорной матрице.

---

## 1. Математические основы

### 1.1 TEON: Mode-1 matricization

Берём моменты двух соседних слоёв одного типа (например, Q из слоя $i$ и $i+1$):

$$\mathbf{M}^{(1)}, \mathbf{M}^{(2)} \in \mathbb{R}^{m \times n}$$

Стакаем в тензор и делаем mode-1 matricization:

$$\mathcal{T} = \text{Ten}(\mathbf{M}^{(1)}, \mathbf{M}^{(2)}) \in \mathbb{R}^{m \times n \times 2}$$

$$\mathbf{Z} = \mathcal{M}_1(\mathcal{T}) = \left[\mathbf{M}^{(1)} \;\Big|\; \mathbf{M}^{(2)}\right] \in \mathbb{R}^{m \times 2n}$$

TEON применяет полную ортогонализацию:

$$\mathbf{O}_{\text{TEON}} = \text{Ortho}(\mathbf{Z}) = \mathbf{U}\mathbf{V}^\top \quad \text{(все } \sigma_i \to 1\text{)}$$

### 1.2 Проблема с tail

SVD матрицы $\mathbf{Z}$:

$$\mathbf{Z} = \sum_{i=1}^{r} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top, \quad r = \min(m, 2n)$$

Авторы Spectra показывают, что top-$k$ направления ($k \approx 1.5\%$ от $r$) содержат spike: $\sigma_1 \gg \sigma_{k+1}$ на 1–2 порядка. Полная ортогонализация TEON ставит $\sigma_i = 1$ для **всех** $i$, то есть усиливает шумные tail-направления.

### 1.3 Spectra-TEON: spike shaping на объединённой матрице

Вместо ортогонализации применяем **spike shrinking**:

**Шаг 1.** Rank-$k$ аппроксимация spike:

$$(\mathbf{U}_k, \mathbf{s}_k, \mathbf{V}_k) = \text{PowerIterSVD}(\mathbf{Z}, k)$$

**Шаг 2.** Остаток (tail):

$$\mathbf{Z}_{\text{tail}} = \mathbf{Z} - \mathbf{U}_k \,\text{diag}(\mathbf{s}_k)\, \mathbf{V}_k^\top$$

**Шаг 3.** Средний масштаб tail:

$$\sigma_{\text{tail}} = \sqrt{\frac{\|\mathbf{Z}_{\text{tail}}\|_F^2}{\min(m, 2n) - k}}$$

**Шаг 4.** Заменяем spike-значения на $\sigma_{\text{tail}}$, tail не трогаем:

$$\mathbf{O} = \mathbf{Z}_{\text{tail}} + \mathbf{U}_k \,\text{diag}(\sigma_{\text{tail}} \cdot \mathbf{1}_k)\, \mathbf{V}_k^\top$$

**Шаг 5.** RMS нормализация шага:

$$\text{RMS} = \frac{\|\mathbf{O}\|_F}{\sqrt{m \cdot 2n}}, \qquad \eta' = \frac{0.2\,\eta}{\text{RMS} + \epsilon}$$

**Шаг 6.** Разбиваем $\mathbf{O}$ обратно и обновляем веса:

$$\mathbf{O}^{(1)}, \mathbf{O}^{(2)} = \text{split}(\mathbf{O}, n, \text{dim}=1)$$

$$\mathbf{W}_t^{(j)} = \mathbf{W}_{t-1}^{(j)} - \eta' \cdot \mathbf{O}^{(j)}, \quad j \in \{1, 2\}$$

### 1.4 Сравнение операций над спектром

| Метод            | Что делает с $\sigma_i$                                                | Tail                    |
| ---------------- | ---------------------------------------------------------------------- | ----------------------- |
| AdamW            | Element-wise нормализация через $\sqrt{v}$                             | Подавляется spike в $v$ |
| Muon/TEON        | $\sigma_i \to 1$ для всех $i$                                          | Усиливается             |
| Spectra          | $\sigma_i^{\text{spike}} \to \sigma_{\text{tail}}$, tail без изменений | Не трогается            |
| **Spectra-TEON** | То же, но на $\mathbf{Z} \in \mathbb{R}^{m \times 2n}$                 | Кросс-слойный spike     |

---

## 2. Реализация

### 2.1 Spike shaping (ядро алгоритма)

```python
import torch
from typing import Optional


def spectra_shape(
    Z: torch.Tensor,
    rank_ratio: float = 0.015,
    n_iter: int = 1,
    V_cache: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Spike-aware спектральное формирование на матрице Z.

    Args:
        Z:           Объединённая матрица m × 2n (после mode-1 matricization)
        rank_ratio:  Доля spike-направлений от min(m, 2n), default 1.5%
        n_iter:      Число power iterations (1 достаточно при warm start)
        V_cache:     Кешированное правое подпространство с прошлого шага

    Returns:
        O:           Shaped матрица того же размера m × 2n
        V_new:       Обновлённый кеш для следующего шага
    """
    m, two_n = Z.shape
    r = min(m, two_n)
    k = max(1, round(rank_ratio * r))

    # --- Power iteration SVD ---
    if V_cache is not None and V_cache.shape == (two_n, k):
        # Warm start: используем кеш с прошлого шага
        V = V_cache
        for _ in range(n_iter):
            P = Z @ V                           # m × k
            U, _ = torch.linalg.qr(P)          # thin QR → U: m × k
            W = Z.T @ U                         # 2n × k
            s = W.norm(dim=0)                   # k
            V = W / (s + 1e-8)                  # 2n × k  (нормализуем столбцы)
        U_k = U
        s_k = s
        V_k = V
    else:
        # Cold start: используем randomized SVD
        U_k, s_k, V_k = torch.svd_lowrank(Z, q=k, niter=2)
        V_k = V_k  # already (2n × k)

    # --- Spike shaping ---
    Z_tail = Z - U_k @ torch.diag(s_k) @ V_k.T

    # Средний масштаб tail
    tail_energy = Z_tail.norm() ** 2
    sigma_tail = (tail_energy / max(r - k, 1)).sqrt()

    # Заменяем spike-значения на sigma_tail, tail без изменений
    O = Z_tail + U_k @ (torch.ones(k, device=Z.device, dtype=Z.dtype) * sigma_tail).diag() @ V_k.T

    return O, V_k.detach()
```

### 2.2 Полный оптимизатор Spectra-TEON

```python
from typing import Callable, Iterable
import torch
from torch import Tensor
from torch.optim import Optimizer


class SpectraTEON(Optimizer):
    """
    Spectra-TEON: кросс-слойный spike-aware оптимизатор.

    Стакает моменты K соседних слоёв одного типа (Q/K/V),
    применяет spike shaping на объединённой матрице,
    разбивает обратно и обновляет веса.

    Args:
        params:       Группы параметров. Каждая группа должна содержать
                      список матриц одного типа (например, все Q-матрицы).
        lr:           Learning rate (будет масштабирован через RMS)
        momentum:     Коэффициент момента (default: 0.95)
        rank_ratio:   Доля spike-направлений (default: 0.015 = 1.5%)
        n_iter:       Число power iterations (default: 1)
        eps:          Для численной стабильности (default: 1e-8)
        K:            Число слоёв в группе (default: 2)
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        momentum: float = 0.95,
        rank_ratio: float = 0.015,
        n_iter: int = 1,
        eps: float = 1e-8,
        K: int = 2,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            rank_ratio=rank_ratio,
            n_iter=n_iter,
            eps=eps,
            K=K,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue

            lr        = group["lr"]
            mu        = group["momentum"]
            rr        = group["rank_ratio"]
            n_iter    = group["n_iter"]
            eps       = group["eps"]
            K         = group["K"]

            # Обрабатываем по K штук за раз
            for i in range(0, len(params), K):
                chunk = params[i : i + K]

                # --- 1. Обновляем моменты ---
                momentums = []
                for p in chunk:
                    state = self.state[p]
                    if "momentum" not in state:
                        state["momentum"] = torch.zeros_like(p)
                        state["V_cache"] = None

                    state["momentum"].mul_(mu).add_(p.grad)
                    momentums.append(state["momentum"])

                # --- 2. Mode-1 matricization: конкатенация по dim=1 ---
                # Каждый момент: m × n → стак: m × (K*n)
                m, n = momentums[0].shape
                Z = torch.cat(momentums, dim=1)  # m × K*n

                # --- 3. Spike shaping вместо ортогонализации ---
                V_cache = self.state[chunk[0]].get("V_cache", None)

                O, V_new = spectra_shape(
                    Z,
                    rank_ratio=rr,
                    n_iter=n_iter,
                    V_cache=V_cache,
                )

                # Сохраняем кеш в первом параметре группы
                self.state[chunk[0]]["V_cache"] = V_new

                # --- 4. RMS нормализация шага ---
                rms = O.norm() / (O.numel() ** 0.5 + eps)
                lr_eff = 0.2 * lr / (rms + eps)

                # --- 5. Split обратно и обновление весов ---
                O_chunks = O.split(n, dim=1)  # K штук по m × n

                for p, O_i in zip(chunk, O_chunks):
                    # Оригинальный TEON scale: sqrt(m/n)
                    # RMS scale уже применён через lr_eff
                    p.add_(O_i, alpha=-lr_eff)

        return loss
```

### 2.3 Утилита для создания групп параметров

```python
def make_teon_groups(
    model: torch.nn.Module,
    lr: float = 1e-3,
    K: int = 2,
    rank_ratio: float = 0.015,
) -> list[dict]:
    """
    Автоматически группирует Q, K, V матрицы трансформера по K соседних слоёв.

    Пример использования:
        optimizer = SpectraTEON(
            make_teon_groups(model, lr=5e-4),
            lr=5e-4
        )
    """
    # Собираем все матрицы по типу
    qkv_by_type: dict[str, list] = {"q": [], "k": [], "v": []}
    other_params: list[Tensor] = []

    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None and param.ndim < 2:
            other_params.append(param)
            continue

        name_lower = name.lower()
        if param.ndim == 2:
            if "q_proj" in name_lower or ".q." in name_lower:
                qkv_by_type["q"].append(param)
            elif "k_proj" in name_lower or ".k." in name_lower:
                qkv_by_type["k"].append(param)
            elif "v_proj" in name_lower or ".v." in name_lower:
                qkv_by_type["v"].append(param)
            else:
                other_params.append(param)
        else:
            other_params.append(param)

    groups = []

    # Для каждого типа (q/k/v) создаём группы по K слоёв
    for mat_type, params in qkv_by_type.items():
        for i in range(0, len(params), K):
            chunk = params[i : i + K]
            groups.append({
                "params": chunk,
                "lr": lr,
                "rank_ratio": rank_ratio,
                "K": len(chunk),
                "name": f"teon_{mat_type}_{i//K}",
            })

    # Остальные параметры — через AdamW
    # (обычно embeddings, norms, biases)
    groups.append({
        "params": other_params,
        "lr": lr * 0.1,  # меньший lr для не-матричных параметров
        "name": "adamw_rest",
        # optimizer для этой группы нужно настроить отдельно
    })

    return groups
```

---

## 3. Интеграция в существующий проект

Если у тебя уже есть TEON с `adamw.py:45`, замена минимальная:

```python
# adamw.py — было (строка ~45):
Z = torch.cat(momentums, dim=1)          # m × 2n
Z_ortho = orthogonalize(Z, ns_steps)     # Polar Express / Newton-Schulz
Q_batch[gi] = Z_ortho.split(n, dim=1)   # split обратно

# adamw.py — стало:
Z = torch.cat(momentums, dim=1)          # m × 2n — не меняется

V_cache = state.get("V_cache", None)     # warm start кеш

O, V_new = spectra_shape(                # spike shaping вместо ortho
    Z,
    rank_ratio=0.015,
    n_iter=1,
    V_cache=V_cache,
)
state["V_cache"] = V_new                 # обновляем кеш

# RMS scale (можно оставить старый (m/n)**0.5 для первого теста)
rms = O.norm() / (O.numel() ** 0.5 + 1e-8)
lr_eff = 0.2 * lr / (rms + 1e-8)

Q_batch[gi] = O.split(n, dim=1)         # split — не меняется
```

---

## 4. Гиперпараметры и рекомендации

### Для 124M (GPT-Small: m=768, n=768)

| Параметр     | Значение       | Обоснование                          |
| ------------ | -------------- | ------------------------------------ |
| `rank_ratio` | `0.015`        | k ≈ 12 из 768 — захватывает spike    |
| `n_iter`     | `1`            | warm start делает 1 итерацию точной  |
| `K`          | `2`            | оптимально по TEON ablation          |
| `lr`         | `5e-4 .. 2e-3` | Spectra более устойчива к большим lr |
| `momentum`   | `0.95`         | стандарт для Muon-семейства          |

### Что попробовать если что-то пошло не так

```
loss не снижается    → увеличь rank_ratio до 0.03 или 0.05
loss нестабилен      → уменьши lr, проверь что V_cache правильно сбрасывается
                       при новой эпохе
медленная сходимость → попробуй убрать RMS scale и вернуть (m/n)**0.5
```

---

## 5. Быстрый тест корректности

```python
def test_spectra_teon():
    torch.manual_seed(42)

    m, n, K = 768, 768, 2
    momentums = [torch.randn(m, n) for _ in range(K)]

    Z = torch.cat(momentums, dim=1)  # m × 2n

    # Проверяем что spike действительно подавляется
    _, s_before, _ = torch.svd_lowrank(Z, q=20, niter=4)

    O, _ = spectra_shape(Z, rank_ratio=0.015, n_iter=2)

    _, s_after, _ = torch.svd_lowrank(O, q=20, niter=4)

    print(f"Top-5 singular values BEFORE: {s_before[:5].tolist()}")
    print(f"Top-5 singular values AFTER:  {s_after[:5].tolist()}")

    # Spike должен уменьшиться, tail — остаться примерно тем же
    assert s_after[0] < s_before[0], "Spike не подавлен!"
    print("✓ Spike подавлен корректно")

    # Размер не изменился
    assert O.shape == Z.shape
    print("✓ Shape сохранён")

if __name__ == "__main__":
    test_spectra_teon()
```

---

## 6. Чего ожидать на 124M

По теории TEON выигрывает у Muon до $\sqrt{K} = \sqrt{2} \approx 1.41\times$ при хорошем выравнивании сингулярных векторов. Spectra выигрывает у AdamW ~30% по wall-clock, у Muon — ниже final loss.

Гибрид потенциально лучше TEON потому что:

- spike на $\mathbf{Z} \in \mathbb{R}^{m \times 2n}$ захватывает **кросс-слойную** общую структуру
- tail не усиливается, в отличие от полной ортогонализации TEON

Прямых экспериментальных данных для этого гибрида нет — это исследовательская территория.
