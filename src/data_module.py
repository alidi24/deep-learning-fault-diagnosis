from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import Dataset, load_dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from src.hf_transforms import SplitToFrame, PCATransform
from src.dataset_manager import DatasetManager


class DataModule(LightningDataModule):
    """
    DataModule for wind turbine bearing dataset with PCA transformation and frame splitting.
    Handles data loading, preprocessing, and augmentation.
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize DataModule with configuration parameters."""
        super().__init__()
        self.config = config
        self.label_map = None
        self.train_val_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> Dataset:
        """
        Load and preprocess the dataset with PCA transformation and frame splitting.
        Returns the processed dataset.
        """
        ds = load_dataset("alidi/wind-turbine-5mw-bearing-dataset")

        ds = ds.map(PCATransform,
                    batched=True,
                    fn_kwargs={
                        'n_components': 1,
                        'input_columns': ['signal_a1a', 'signal_a2a'],
                        'output_column': 'pc_feature_1',
                        'standardize': True
                    }
        )

        self.ds = ds.map(SplitToFrame,
                    batched=True, 
                    fn_kwargs={"frame_length": self.config["frame_length"], 
                               "hop_length": self.config["hop_length"],
                               "signal_column": "pc_feature_1"}
        )

        return self.ds

    def setup(self, stage: Optional[str] = None) -> Union[Tuple[Dataset, Dataset], Dataset]:
        """
        Set up train, validation, and test datasets based on stage.
        Prints dataset sizes for monitoring.
        """
        if stage == "fit" or stage is None:
            self.train_dataset = DatasetManager.create_training_data(
                self.ds,
                self.config,
            )
            print(f"Training dataset size: {len(self.train_dataset)}")

            self.val_dataset = DatasetManager.create_validation_data(
                self.ds, 
                self.config
            )
            print(f"Validation dataset size: {len(self.val_dataset)}")

            return self.train_dataset, self.val_dataset

        if stage == "test":
            self.test_dataset = DatasetManager.create_test_data(
                self.ds, 
                self.config
            )
            print(f"Test dataset size: {len(self.test_dataset)}")
            return self.test_dataset

    def train_dataloader(self) -> DataLoader:
        """
        Returns DataLoader for training data with shuffling and augmentation options.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns DataLoader for validation data."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size"],
            drop_last=True,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Returns DataLoader for test data."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.config["batch_size"],
            drop_last=False,
            collate_fn=self._collate_fn,
        )

    def train_collate(self, batch):
        """Wrapper for collate function with training-specific handling."""
        return self._collate_fn(batch, is_train=True)

    def _collate_fn(
        self, batch: List[Dict[str, Any]], is_train: bool = False
    ) -> torch.Tensor:
        """
        Collates batch data into tensors. Applies augmentation during training if configured.
        Returns: Tensor of shape [batch_size, 1, signal_length]
        """
        if is_train and self.config["is_aug"]:
            batch = self.__augmentation(batch)

        default_float_dtype = torch.get_default_dtype()
        tensor_signals = [
            torch.tensor(sample["pc_feature_1"], dtype=default_float_dtype).unsqueeze(0) 
            for sample in batch
        ]

        return torch.stack(tensor_signals)

    def __augmentation(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Applies random augmentations to the signal:
        - Signal inversion (70% chance)
        - Signal rolling (70% chance)
        - Gaussian noise addition (80% chance)
        """
        transformed_batch = []
        for example in batch:
            signal = np.array(example["pc_feature_1"])
            
            if np.random.rand() > 0.3:
                signal = -signal

            if np.random.rand() > 0.3:
                roll_amount = np.random.randint(1, len(signal))
                direction = np.random.choice(["start", "end"])
                signal = np.roll(signal, roll_amount if direction == "start" else -roll_amount)

            if np.random.rand() > 0.2:
                signal_std = np.std(signal)
                noise_level = np.random.uniform(0.05, 0.10)
                noise = np.random.normal(0, signal_std * noise_level, size=signal.shape)
                signal = signal + noise

            transformed_example = example.copy()
            transformed_example["pc_feature_1"] = signal
            transformed_batch.append(transformed_example)
        return transformed_batch