from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from src.datasets.base_data import ICIDataset


def test_all_cell_mean_uses_every_donor_cell(tmp_path: Path) -> None:
    fold_dir = tmp_path / "SEED42" / "CV0"
    fold_dir.mkdir(parents=True)
    features = torch.tensor(
        [
            [1.0, 3.0],
            [3.0, 5.0],
            [100.0, 200.0],
        ]
    )
    torch.save(features, fold_dir / "val_hvg.pt")
    pd.DataFrame(
        {
            "donor_id": ["a", "a", "b"],
            "Response": ["NR", "NR", "R"],
        }
    ).to_csv(fold_dir / "val_donor_info.csv", index=False)

    dataset = ICIDataset(
        cv=0,
        state="val",
        root_dir=str(tmp_path),
        seed=42,
        target_cells=1,
        all_cell_mean=True,
    )

    donor_a, label_a = dataset[0]
    donor_b, label_b = dataset[1]
    assert donor_a.shape == (1, 2)
    assert torch.equal(donor_a, torch.tensor([[2.0, 4.0]]))
    assert torch.equal(donor_b, torch.tensor([[100.0, 200.0]]))
    assert label_a.item() == 0
    assert label_b.item() == 1
