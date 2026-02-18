# AdaPlus Optimizer

## Ключевая идея

AdaPlus = AdamW + Nesterov momentum (из Nadam) + precise stepsize adjustment (из AdaBelief).

Не вводит дополнительных гиперпараметров по сравнению с AdamW.

## Математические формулы

### 1. Gradient & Weight Decay

$$g_t = \nabla_\theta f_t(\theta_{t-1})$$

$$\theta_t = \theta_{t-1} - \gamma \lambda \theta_{t-1} \quad \text{(decoupled weight decay)}$$

### 2. First moment (EMA of gradients)

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

### 3. Second moment — "belief" (отличие от Adam/AdamW!)

Вместо $v_t = \text{EMA}(g_t^2)$ используется:

$$s_t = \beta_2 s_{t-1} + (1 - \beta_2)(g_t - m_t)^2 + \epsilon$$

> **Это ключевое отличие**: $s_t$ отслеживает отклонение градиента от его EMA-предсказания.
> Когда градиент совпадает с предсказанием ($g_t \approx m_t$), $s_t$ мал → шаг увеличивается.
> Когда градиент сильно отклоняется, $s_t$ велик → шаг уменьшается.

### 4. Nesterov momentum (отличие от AdamW!)

Вместо обычного $m_t$ используется look-ahead:

$$\bar{m}_t = \beta_1 m_t + (1 - \beta_1) g_t$$

### 5. Bias correction

$$\hat{m}_t = \frac{\bar{m}_t}{1 - \beta_1^t}, \quad \hat{s}_t = \frac{s_t}{1 - \beta_2^t}$$

### 6. Parameter update

$$\theta_t = \theta_{t-1} - \frac{a \cdot \hat{m}_t}{\sqrt{\hat{s}_t} + \epsilon}$$

---

## Сравнение update rules

| Optimizer   | Update direction (знаменатель)                                                        |
| ----------- | ------------------------------------------------------------------------------------- |
| AdamW       | $\sqrt{\hat{v}_t} + \epsilon$ где $v_t = \text{EMA}(g_t^2)$                           |
| AdaBelief   | $\sqrt{\hat{s}_t} + \epsilon$ где $s_t = \text{EMA}((g_t - m_t)^2)$                   |
| Nadam       | $\sqrt{\hat{v}_t} + \epsilon$ но с Nesterov $\bar{m}_t$ в числителе                   |
| **AdaPlus** | $\sqrt{\hat{s}_t} + \epsilon$ **+ Nesterov** $\bar{m}_t$ **+ decoupled weight decay** |

---

## Реализация на PyTorch

```python
import math
import torch
from torch.optim import Optimizer


class AdaPlus(Optimizer):
    """AdaPlus: Nesterov momentum + precise stepsize adjustment on AdamW basis.

    Args:
        params: iterable of parameters to optimize
        lr: learning rate (default: 1e-3)
        betas: coefficients for computing running averages (default: (0.9, 0.999))
        eps: term added to denominator for numerical stability (default: 1e-8)
        weight_decay: decoupled weight decay factor (default: 1e-2)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdaPlus does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)  # first moment (EMA of grad)
                    state["s"] = torch.zeros_like(p)  # second moment (EMA of (grad - m)^2)

                m, s = state["m"], state["s"]
                state["step"] += 1
                t = state["step"]

                # --- Step 1: Decoupled weight decay (AdamW-style) ---
                if weight_decay != 0:
                    p.mul_(1.0 - lr * weight_decay)

                # --- Step 2: Update first moment (EMA of gradients) ---
                # m_t = β1 * m_{t-1} + (1 - β1) * g_t
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # --- Step 3: Update second moment — "belief" term ---
                # s_t = β2 * s_{t-1} + (1 - β2) * (g_t - m_t)^2 + eps
                # Note: eps is added INSIDE the EMA (как в AdaBelief), это важно
                s.mul_(beta2).addcmul_(grad - m, grad - m, value=1.0 - beta2).add_(eps)

                # --- Step 4: Nesterov momentum ---
                # m_bar_t = β1 * m_t + (1 - β1) * g_t
                m_bar = beta1 * m + (1.0 - beta1) * grad

                # --- Step 5: Bias correction ---
                bias_correction1 = 1.0 - beta1**t
                bias_correction2 = 1.0 - beta2**t

                m_hat = m_bar / bias_correction1
                s_hat = s / bias_correction2

                # --- Step 6: Parameter update ---
                # θ_t = θ_{t-1} - lr * m_hat / (sqrt(s_hat) + eps)
                p.addcdiv_(m_hat, s_hat.sqrt().add_(eps), value=-lr)

        return loss
```

## Пример использования

```python
model = MyModel()
optimizer = AdaPlus(model.parameters(), lr=1e-3, weight_decay=1e-2)

for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(batch.x), batch.y)
        loss.backward()
        optimizer.step()
```

## Рекомендуемые гиперпараметры (из статьи)

| Задача                                 | lr   | β1               | β2    | ε     | weight_decay |
| -------------------------------------- | ---- | ---------------- | ----- | ----- | ------------ |
| Image Classification (VGG)             | 1e-3 | 0.9              | 0.999 | 1e-8  | 1e-2         |
| Image Classification (ResNet/DenseNet) | 1e-2 | 0.9              | 0.999 | 1e-8  | 1e-2         |
| Language Modeling (LSTM)               | 1e-3 | 0.9              | 0.999 | 1e-16 | 1e-2         |
| GAN Training                           | 2e-4 | search [0.5–0.9] | 0.999 | 1e-12 | 1e-2         |

## Важные нюансы реализации

1. **eps внутри EMA**: В строке `s_t` epsilon добавляется _внутри_ скользящего среднего (строка 7 алгоритма), а не только в знаменателе при делении. Это отличает AdaPlus/AdaBelief от Adam.

2. **Weight decay decoupled**: Применяется как `p *= (1 - lr * λ)` _до_ градиентного шага, не через L2-регуляризацию.

3. **Nesterov look-ahead**: `m_bar = β1 * m_t + (1 - β1) * g_t` — используется текущий `m_t`, а не `m_{t-1}`, что эквивалентно подстановке следующего шага момента.
