# NAMO: Adam Improves Muon — Adaptive Moment Estimation with Orthogonalized Momentum

**Paper:** Zhang, Liu, Schaeffer (UCLA), arXiv:2602.17080
**Code:** https://github.com/minxin-zhg/namo

## TL;DR

NAMO и NAMO-D — первая теоретически обоснованная интеграция ортогонализированного моментума (Muon) с Adam-style адаптацией шума. Ключевая идея: масштабировать ортогонализированное обновление адаптивным скаляром (NAMO) или диагональной матрицей (NAMO-D), чтобы уменьшить эффективный stepsize когда градиенты шумные или итерации приближаются к стационарной точке.

## Алгоритмы

### NAMO (скалярная адаптация)

Поддерживает EMA квадрата нормы Фробениуса градиента (скаляр `v_t`), формирует адаптивный скаляр α_t:

```
M_t = μ₁ M_{t-1} + (1 - μ₁) G_t         # EMA momentum (как Muon)
v_t = μ₂ v_{t-1} + (1 - μ₂) ||G_t||²_F  # EMA squared Frobenius norm (СКАЛЯР)
O_t = Orth(M_t)                            # Newton-Schulz/Polar Express
α_t = (√(1-μ₂ᵗ)/(1-μ₁ᵗ)) · ||M_t||_F / (√v_t + ε)  # adaptive scalar
Θ_t = Θ_{t-1} - η · α_t · O_t            # update
```

**Дополнительная стоимость:** O(mn) — одна Frobenius norm, пренебрежимо мало поверх Muon.
**Дополнительная память:** один float32 скаляр `v_t` на параметр — почти ноль.

**Свойство (Lemma 1):** α_t ≤ √((1-μ₁)/(1-μ₂)), т.е. адаптивный скаляр ограничен сверху.

### NAMO-D (диагональная/поколоночная адаптация)

Поддерживает EMA квадратов norm колонок градиента (вектор `v_t ∈ R^n`):

```
M_t = μ₁ M_{t-1} + (1 - μ₁) G_t
v_t[j] = μ₂ v_{t-1}[j] + (1 - μ₂) ||G_t[:,j]||²   # per-column EMA
O_t = Orth(M_t)
d_t[j] = (√(1-μ₂ᵗ)/(1-μ₁ᵗ)) · ||M_t[:,j]|| / (√v_t[j] + ε)  # per-column scale
d̄_t = mean(d_t)
d̃_t[j] = clamp(d_t[j], c · d̄_t, d̄_t / c)         # clamp toward mean
D_t = diag(d̃_t)
Θ_t = Θ_{t-1} - η · O_t · D_t                       # RIGHT-multiply by diagonal
```

**Clamping c ∈ (0, 1]:** контролирует баланс между neuron-wise адаптацией (малый c) и сохранением ортогональности (c → 1, D_t → scalar·I).

**Дополнительная память:** вектор `v_t ∈ R^n` на параметр (n — число колонок/нейронов).

### Weight Decay (важный нюанс!)

NAMO и NAMO-D применяют адаптивный скаляр/диагональ также и к weight decay:

- **NAMO:**  `Θ_t = Θ_{t-1} - η · α_t · (O_t + λ · Θ_{t-1})`
- **NAMO-D:** `Θ_t = Θ_{t-1} - η · (O_t + λ · Θ_{t-1}) · D_t`

Это отличается от Muon, где weight decay имеет фиксированный rate.

## Теоретические гарантии

| Setting | NAMO | NAMO-D | Optimal |
|---------|------|--------|---------|
| Deterministic | O(T^{-1/2}) | O(T^{-1/2}) | O(T^{-1/2}) ✓ |
| Stochastic | O(T^{-1/4} + σ^{1/2} b^{-1/4} T^{-1/8}) | то же | O(T^{-1/4}) при b = Ω(σ² T^{1/2}) |

Оба алгоритма достигают оптимальных rates — ортогонализация + адаптивное масштабирование не ухудшают сходимость.

## Экспериментальные результаты (GPT-2 на OpenWebText)

### GPT-2 124M (50K steps)

| Optimizer | Val Loss | LR |
|-----------|----------|-----|
| AdamW     | 3.0643   | 0.0013 |
| Muon      | 3.0435   | 0.0013 |
| **NAMO**  | 3.0351   | **0.012** |
| **NAMO-D**| **3.0246** | **0.009**, c=0.1 |

### GPT-2 355M (10K steps)

| Optimizer | Val Loss | LR |
|-----------|----------|-----|
| AdamW     | 2.9914   | 0.0009 |
| Muon      | 2.9684   | 0.0009 |
| **NAMO**  | 2.9516   | **0.007** |
| **NAMO-D**| **2.9507** | **0.009**, c=0.9 |

**Ключевые наблюдения:**
1. NAMO/NAMO-D используют значительно более высокие LR (10x vs Muon/AdamW)
2. NAMO-D лучше NAMO при длинном обучении (50K steps)
3. Оптимальный c зависит от масштаба модели (0.1 для 124M, 0.9 для 355M)
4. Более широкий диапазон рабочих LR (tuning robustness)

### Гиперпараметры

| Param | NAMO | NAMO-D |
|-------|------|--------|
| μ₁ (momentum) | 0.95 | 0.95 |
| μ₂ (v EMA) | 0.99 | 0.99 |
| ε | 1e-8 | 1e-8 |
| c (clamp) | — | 0.1–0.9 |
| λ (weight decay) | 0.01 | 0.01 |

## Применимость к nash-llm: T-NAMO (TEON + NAMO)

### Ответ на главный вопрос: Можно ли?

**Да, NAMO и NAMO-D можно интегрировать с TEON.** Ортогонализация в NAMO/NAMO-D — это отдельный шаг от адаптивного масштабирования. TEON заменяет только способ ортогонализации (stacking K слоёв → joint Orth), а адаптивное масштабирование NAMO/NAMO-D применяется после.

### Архитектура T-NAMO

```
# T-NAMO = TEON stacking + NAMO adaptive scalar

# 1. Per-param: EMA momentum + EMA v (как обычно)
M_t^(k) = μ₁ M_{t-1}^(k) + (1-μ₁) G_t^(k)     для k=1,...,K
v_t^(k) = μ₂ v_{t-1}^(k) + (1-μ₂) ||G_t^(k)||²_F

# 2. TEON stacking: joint ортогонализация
Z_t = [M_t^(1) | M_t^(2) | ... | M_t^(K)]       R^{m × nK}
Q_t = Orth(Z_t)                                   R^{m × nK}
O_t^(k) = Q_t[:, k*n:(k+1)*n]                    split back

# 3. NAMO adaptive scalar (per-param)
α_t^(k) = bc_factor · ||M_t^(k)||_F / (√v_t^(k) + ε)

# 4. Update
W_t^(k) = W_{t-1}^(k) - lr · α_t^(k) · scale · O_t^(k)
```

### Архитектура T-NAMO-D

```
# T-NAMO-D = TEON stacking + NAMO-D column-wise scaling

# 1. Per-param: EMA momentum + column-wise EMA v
M_t^(k) = μ₁ M_{t-1}^(k) + (1-μ₁) G_t^(k)
v_t^(k)[j] = μ₂ v_{t-1}^(k)[j] + (1-μ₂) ||G_t^(k)[:,j]||²   ∈ R^n

# 2. TEON stacking → joint orth → split back
Z_t = [M_t^(1) | ... | M_t^(K)]
Q_t = Orth(Z_t)
O_t^(k) = Q_t[:, k*n:(k+1)*n]

# 3. NAMO-D per-column scaling (per-param, after split)
d_t^(k)[j] = bc_factor · ||M_t^(k)[:,j]|| / (√v_t^(k)[j] + ε)
D_t^(k) = diag(clamp(d_t^(k), ...))

# 4. Update: right-multiply by diagonal
W_t^(k) = W_{t-1}^(k) - lr · scale · O_t^(k) · D_t^(k)
```

### Ключевые решения при интеграции

#### 1. Где вычислять адаптивный скаляр: до или после TEON stacking?

**Рекомендация: per-param (до stacking)**

Вычислять α_t^(k) или D_t^(k) для каждого параметра индивидуально, используя его собственные M_t^(k) и G_t^(k). Причины:
- v_t должен отражать шум конкретного слоя, а не усреднённый шум K слоёв
- Теоретические гарантии NAMO предполагают per-parameter v_t
- TEON отвечает за направление (joint orth), NAMO за масштаб — ортогональные роли

Альтернатива (joint v_t по Z): возможна, но теряет per-layer noise adaptation.

#### 2. MUON params: обычный NAMO

Для out_proj, fc1, fc2 — стандартный NAMO/NAMO-D без stacking (как сейчас per-layer Muon, просто с добавлением adaptive scaling).

#### 3. Nesterov momentum

Референсная реализация NAMO использует Nesterov momentum:
```python
buf.lerp_(g, 1 - momentum)       # M_t = μ M_{t-1} + (1-μ) G_t
g_for_orth = g.lerp(buf, momentum)  # Nesterov: G_t + μ · M_t (blend)
```
Это может дать дополнительный выигрыш. В текущем Muon nash-llm используется обычный momentum.

#### 4. LR scaling

Референсная реализация NAMO использует `scale_coeff * sqrt(max(m, n))` вместо нашего `sqrt(m/n)`. Это существенное различие — нужно выбрать:
- Наш `sqrt(m/n)` — консервативнее, проверен с текущими configs
- Их `scale_coeff * sqrt(max(m,n))` — агрессивнее, но оптимальные LR пересвипованы под эту формулу

**Рекомендация:** оставить наш `sqrt(m/n)` и подобрать LR независимо.

### Изменения в коде

**Файлы для модификации:**
1. `nash_llm/optim/muon.py` — добавить NAMO variant в класс Muon или создать новый класс
2. `nash_llm/optim/adamw.py` — добавить параметры mu2, eps, clamp_c в configure_optimizers
3. `nash_llm/config/config.py` — добавить поля: `muon_mu2`, `muon_eps`, `namo_clamp_c`, `optimizer_variant` (enum: muon/namo/namo_d)
4. `nash_llm/training/trainer.py` — минимальные изменения (прокидывание конфига)

**Дополнительная память T-NAMO vs текущий Muon:**
- NAMO: +1 float32 скаляр на parametr → пренебрежимо мало
- NAMO-D: +1 float32 вектор [n] на параметр → ~n×4 bytes (для 768-dim → 3KB на параметр, всего ~200KB для 100M модели)

**Дополнительные FLOPS:**
- NAMO: O(mn) для Frobenius norm — ничтожно vs ортогонализация O(mn² · steps)
- NAMO-D: O(mn) для column norms + O(n) для clamping — тоже ничтожно

### Конкретный план эксперимента

1. **Начать с NAMO** (проще, нет гиперпараметра c):
   - μ₁ = 0.95 (как текущий momentum)
   - μ₂ = 0.99
   - ε = 1e-8
   - Свипнуть LR: попробовать текущий muon_lr * {1, 3, 5, 10}

2. **Затем NAMO-D:**
   - Те же μ₁, μ₂, ε
   - c = 0.1 для маленьких моделей, c = 0.5–0.9 для больших
   - Свипнуть LR и c

3. **Сравнить T-NAMO vs TEON+Muon** на pretrain_small.yaml (100M model, 10M tokens):
   - Метрика: val_loss, скорость сходимости
   - При успехе → масштабировать на pretrain_1b.yaml

### Потенциальные риски

1. **LR sensitivity:** NAMO работает на значительно более высоких LR (10x). Нужен пересвип.
2. **Clamping c sensitivity:** для NAMO-D оптимальный c сильно зависит от масштаба модели.
3. **Nesterov vs vanilla momentum:** референсная реализация использует Nesterov, наш Muon — нет. Можно добавить как опцию.
4. **Weight decay coupling:** NAMO применяет α_t к weight decay, наш Muon — нет. Это изменяет effective WD rate.

## Связь с другими работами

- **AdaMuon, NorMuon:** похожие идеи, но без теоретических гарантий
- **PRISM:** Adam-style preconditioner для Muon, но дороже вычислительно
- **DeVA:** Kronecker preconditioner, высокие overhead
- **NAMO** — самый лёгкий и теоретически обоснованный вариант
