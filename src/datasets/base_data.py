from pathlib import Path

import torch
import pandas as pd
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ICIDataset(Dataset):
    def __init__(
        self,
        cv,
        state,
        root_dir='data/ICI_CVOnly_scConcept_512',
        seed=42,
        target_cells=1000,
        all_cell_mean=False,
    ):
        """
        Args:
            cv (int/str): Cross-validation fold (e.g., 0).
            state (str): 상태 (e.g., 'train', 'test').
            root_dir (str): 데이터 파일이 위치한 디렉토리 경로.
            seed (int): 랜덤 시드 값.
        """
        self.cv = cv
        self.state = state
        root_path = Path(root_dir).expanduser()
        if not root_path.is_absolute():
            root_path = PROJECT_ROOT / root_path
        self.root_dir = root_path.resolve()
        self.seed = seed
        self.donor_col = 'donor_id'
        self.target_col = 'Response' # 타겟 컬럼명 지정
        self.target_cells = target_cells
        self.all_cell_mean = bool(all_cell_mean)
        
        # 1. 파일 경로 구성
        if state == 'external':
            hvg_path = self.root_dir.parent / "ICI_GSE285888_scConcept_512.pt"
            info_path = self.root_dir.parent / "ICI_GSE285888_scConcept_512_info.csv"
        elif state == 'test':
            hvg_path = self.root_dir / f"SEED{seed}" / f"{state}_hvg.pt"
            info_path = self.root_dir / f"SEED{seed}" / f"{state}_donor_info.csv"
        else:
            hvg_path = self.root_dir / f"SEED{seed}" / f"CV{cv}" / f"{state}_hvg.pt"
            info_path = self.root_dir / f"SEED{seed}" / f"CV{cv}" / f"{state}_donor_info.csv"
        
        # 2. 데이터 로드
        if not hvg_path.is_file() or not info_path.is_file():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: \n{hvg_path} \n{info_path}")
        self.cell_features = torch.load(hvg_path)
        self.donor_info = pd.read_csv(info_path)
        if len(self.cell_features) != len(self.donor_info):
            raise ValueError(f"데이터 길이 불일치: HVG({len(self.cell_features)}) vs Info({len(self.donor_info)})")
        if self.donor_col not in self.donor_info.columns:
            raise ValueError(f"CSV에 '{self.donor_col}' 컬럼이 없습니다.")
        if self.target_col not in self.donor_info.columns:
            raise ValueError(f"CSV에 '{self.target_col}' 컬럼이 없습니다.")

        # 3. Donor별 인덱스 그룹화 및 라벨 매핑
        self.unique_donors = self.donor_info[self.donor_col].unique()
        self.donor_to_indices = self.donor_info.groupby(self.donor_col).indices
        self.label_map = {'NR': 0, 'R': 1}
        
        donor_response_df = self.donor_info[[self.donor_col, self.target_col]].drop_duplicates()
        self.donor_to_label_str = dict(zip(donor_response_df[self.donor_col], donor_response_df[self.target_col]))

    def __len__(self):
        return len(self.unique_donors)

    def __getitem__(self, idx):
        """
        Returns:
            bag_features (Tensor): (1000, N_features)
            label (LongTensor): 0 (NR) or 1 (R) - 스칼라 텐서
        """
        donor_id = self.unique_donors[idx]
        
        # 1. Features 추출 (전체 평균 또는 샘플링 및 패딩)
        all_indices = self.donor_to_indices[donor_id]
        num_cells = len(all_indices)
        if self.all_cell_mean:
            # Legacy fast path for explicitly requested mean-only evaluation.
            # Architecture v12 normally receives all variable-length cells.
            bag_features = self.cell_features[all_indices].mean(dim=0, keepdim=True)
        else:
            target_size = self.target_cells
            if target_size == -1:
                bag_features = self.cell_features[all_indices]
            elif num_cells >= target_size:
                if self.state == 'train':
                    selected_indices = torch.randperm(num_cells)[:target_size]
                else:
                    selected_indices = torch.arange(target_size)
                actual_indices = all_indices[selected_indices]
                bag_features = self.cell_features[actual_indices]
            else:
                bag_features = self.cell_features[all_indices]
                padding_size = target_size - num_cells
                padding = torch.zeros((padding_size, bag_features.shape[1]), dtype=bag_features.dtype)
                bag_features = torch.cat([bag_features, padding], dim=0)

        # 2. Label 추출 및 변환
        label_str = self.donor_to_label_str[donor_id]
        
        # 예외 처리: NR/R 이외의 값이 있을 경우
        if label_str not in self.label_map:
             raise ValueError(f"알 수 없는 라벨 값입니다: {label_str} (Donor: {donor_id})")
             
        label = self.label_map[label_str]
        
        return bag_features, torch.tensor(label, dtype=torch.long)
