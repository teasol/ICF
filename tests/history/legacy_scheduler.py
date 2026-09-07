import unittest

import torch

from src.utils.schedulers import WarmupReduceLROnPlateau


class WarmupReduceLROnPlateauTest(unittest.TestCase):
    def test_holds_target_lr_until_plateau_start(self) -> None:
        parameter = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        scheduler = WarmupReduceLROnPlateau(
            optimizer,
            warmup_epochs=2,
            warmup_start_factor=0.1,
            plateau_start_epoch=4,
            mode="min",
            factor=0.5,
            patience=0,
        )

        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.1)
        scheduler.step(4.0)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.55)
        scheduler.step(3.0)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1.0)
        scheduler.step(2.0)
        scheduler.step(1.0)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1.0)

        scheduler.step(1.0)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1.0)
        scheduler.step(2.0)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.5)

    def test_rejects_plateau_before_warmup_end(self) -> None:
        parameter = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        with self.assertRaises(ValueError):
            WarmupReduceLROnPlateau(
                optimizer,
                warmup_epochs=10,
                plateau_start_epoch=9,
            )


if __name__ == "__main__":
    unittest.main()
