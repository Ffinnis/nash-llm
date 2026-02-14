from nash_llm.training.scheduler import CosineScheduler


class TestCosineScheduler:
    def test_warmup_phase(self):
        scheduler = CosineScheduler(max_lr=1e-3, min_lr=1e-5, warmup_steps=100, max_steps=1000)
        lr_0 = scheduler.get_lr(0)
        lr_50 = scheduler.get_lr(50)
        lr_100 = scheduler.get_lr(100)
        assert lr_0 < lr_50 < lr_100
        assert abs(lr_100 - 1e-3) < 1e-8

    def test_cosine_decay(self):
        scheduler = CosineScheduler(max_lr=1e-3, min_lr=1e-5, warmup_steps=100, max_steps=1000)
        lr_100 = scheduler.get_lr(100)
        lr_500 = scheduler.get_lr(500)
        lr_999 = scheduler.get_lr(999)
        assert lr_100 > lr_500 > lr_999

    def test_min_lr_at_end(self):
        scheduler = CosineScheduler(max_lr=1e-3, min_lr=1e-5, warmup_steps=100, max_steps=1000)
        lr_end = scheduler.get_lr(1000)
        assert abs(lr_end - 1e-5) < 1e-8

    def test_beyond_max_steps(self):
        scheduler = CosineScheduler(max_lr=1e-3, min_lr=1e-5, warmup_steps=100, max_steps=1000)
        lr = scheduler.get_lr(2000)
        assert abs(lr - 1e-5) < 1e-8
