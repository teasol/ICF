import tempfile
import unittest
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.utils.utils import AlwaysSaveLastModelCheckpoint


class _WorseningValidationModule(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def training_step(self, batch, batch_idx):
        return self.weight.square()

    def validation_step(self, batch, batch_idx):
        value = torch.tensor(float(self.current_epoch), device=self.device)
        self.log("val_loss", value, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


class AlwaysSaveLastModelCheckpointTest(unittest.TestCase):
    def test_last_checkpoint_advances_when_metric_does_not_improve(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            callback = AlwaysSaveLastModelCheckpoint(
                dirpath=directory,
                filename="{epoch:03d}-{val_loss:.4f}",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                save_last=True,
            )
            loader = DataLoader(TensorDataset(torch.ones(1, 1)), batch_size=1)
            trainer = L.Trainer(
                accelerator="cpu",
                devices=1,
                max_epochs=3,
                logger=False,
                callbacks=[callback],
                enable_checkpointing=True,
                enable_model_summary=False,
                enable_progress_bar=False,
                limit_train_batches=1,
                limit_val_batches=1,
                num_sanity_val_steps=0,
            )
            trainer.fit(
                _WorseningValidationModule(),
                train_dataloaders=loader,
                val_dataloaders=loader,
            )

            checkpoint = torch.load(
                Path(directory) / "last.ckpt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(checkpoint["epoch"], 2)
            self.assertEqual(callback.best_model_score.item(), 0.0)

    def test_checkpoint_selection_ignores_ineligible_early_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            callback = AlwaysSaveLastModelCheckpoint(
                dirpath=directory,
                filename="{epoch:03d}-{val_loss:.4f}",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                save_last=False,
                selection_start_epoch=2,
            )
            loader = DataLoader(TensorDataset(torch.ones(1, 1)), batch_size=1)
            trainer = L.Trainer(
                accelerator="cpu",
                devices=1,
                max_epochs=3,
                logger=False,
                callbacks=[callback],
                enable_checkpointing=True,
                enable_model_summary=False,
                enable_progress_bar=False,
                limit_train_batches=1,
                limit_val_batches=1,
                num_sanity_val_steps=0,
            )
            trainer.fit(
                _WorseningValidationModule(),
                train_dataloaders=loader,
                val_dataloaders=loader,
            )

            self.assertEqual(callback.best_model_score.item(), 2.0)
            self.assertIn("epoch=002", callback.best_model_path)


if __name__ == "__main__":
    unittest.main()
