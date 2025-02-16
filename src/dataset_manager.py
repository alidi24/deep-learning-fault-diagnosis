from typing import Dict, Any, List, Tuple
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

class DatasetManager:
    @staticmethod
    def _split_by_label(dataset: Dataset, test_size: float, random_seed: int = 42) -> Tuple[Dataset, Dataset]:
        """
        Split dataset while maintaining label distribution
        
        Args:
            dataset: Input dataset
            test_size: Proportion of data to include in the test split
            random_seed: Random seed for reproducibility
        """
        # Get unique labels
        labels = dataset.unique("label")
        
        # Initialize lists for train and test indices
        train_indices = []
        test_indices = []
        
        # Split each label's data separately
        for label in labels:
            # Get indices for current label
            label_indices = [i for i, l in enumerate(dataset["label"]) if l == label]
            
            # Split indices
            label_train_indices, label_test_indices = train_test_split(
                label_indices,
                test_size=test_size,
                random_seed=random_seed
            )
            
            train_indices.extend(label_train_indices)
            test_indices.extend(label_test_indices)
        
        # Create new datasets using the indices
        train_dataset = dataset.select(train_indices)
        test_dataset = dataset.select(test_indices)
        
        return train_dataset, test_dataset

    @classmethod
    def create_train_val_test_split(
        cls, 
        dataset: Dataset, 
        test_size: float,
        val_size: float,
        random_seed: int = 42
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Create train, validation, and test splits while maintaining label distribution
        
        Args:
            dataset: Input dataset
            test_size: Proportion of data to include in the test split (e.g., 0.2 for 20%)
            val_size: Proportion of remaining data (after test split) for validation
            random_seed: Random seed for reproducibility
        """
        # First split: train_val and test
        train_val_dataset, test_dataset = cls._split_by_label(
            dataset, 
            test_size=test_size, 
            random_seed=random_seed
        )
        
        # Second split of train_val: train and val
        train_dataset, val_dataset = cls._split_by_label(
            train_val_dataset,
            test_size=val_size,
            random_seed=random_seed
        )
        
        return train_dataset, val_dataset, test_dataset

    @classmethod
    def create_training_data(cls, dataset: Dataset, config: Dict[str, Any]) -> Dataset:
        """
        Create the training dataset
        
        Args:
            dataset: Input dataset
            config: Configuration dictionary containing 'test_size' and 'val_size'
        """
        test_size = config.get('test_size', 0.2)
        val_size = config.get('val_size', 0.2)
        random_seed = config.get('random_seed', 42)
        
        train_dataset, _, _ = cls.create_train_val_test_split(
            dataset,
            test_size=test_size,
            val_size=val_size,
            random_seed=random_seed
        )
        return train_dataset

    @classmethod
    def create_validation_data(cls, dataset: Dataset, config: Dict[str, Any]) -> Dataset:
        """
        Create the validation dataset
        
        Args:
            dataset: Input dataset
            config: Configuration dictionary containing 'test_size' and 'val_size'
        """
        test_size = config.get('test_size', 0.2)
        val_size = config.get('val_size', 0.2)
        random_seed = config.get('random_seed', 42)
        
        _, val_dataset, _ = cls.create_train_val_test_split(
            dataset,
            test_size=test_size,
            val_size=val_size,
            random_seed=random_seed
        )
        return val_dataset

    @classmethod
    def create_test_data(cls, dataset: Dataset, config: Dict[str, Any]) -> Dataset:
        """
        Create the test dataset
        
        Args:
            dataset: Input dataset
            config: Configuration dictionary containing 'test_size' and 'val_size'
        """
        test_size = config.get('test_size', 0.2)
        val_size = config.get('val_size', 0.2)
        random_seed = config.get('random_seed', 42)
        
        _, _, test_dataset = cls.create_train_val_test_split(
            dataset,
            test_size=test_size,
            val_size=val_size,
            random_seed=random_seed
        )
        return test_dataset