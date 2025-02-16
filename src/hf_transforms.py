
import librosa
import numpy as np
from sklearn.decomposition import PCA


class PCATransform:
    def __call__(
        self, 
        examples: dict[str, list],
        n_components: int = 2,
        input_columns: list[str] = None,
        output_column: str = "pca_features",
        standardize: bool = True
    ) -> dict[str, list]:
        """Apply PCA transformation to the input examples.
        
        Args:
            examples (dict[str, list]): Input examples with the specified columns
            n_components (int): Number of components for PCA
            input_columns (list[str]): List of column names to apply PCA on
            output_column (str): Name of the output column
            standardize (bool): Whether to standardize the input data
            
        Returns:
            dict[str, list]: Updated examples with PCA output
        """
        if input_columns is None:
            raise ValueError("input_columns must be specified")
        
        if len(input_columns) < 2:
            raise ValueError("At least two input columns must be specified")
            
        pca = PCA(n_components=n_components)
        
        # Initialize output column
        examples[output_column] = [None] * len(examples[input_columns[0]])
        
        # Process each sample
        for ind in range(len(examples[input_columns[0]])):
            # Get data from all input columns for this sample
            sample_data = [examples[col][ind] for col in input_columns]
            data = np.stack(sample_data, axis=1)
            
            # Standardize if requested
            if standardize:
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                data = (data - mean) / (std + 1e-8)
            
            # Apply PCA transformation
            transformed_data = pca.fit_transform(data)
            
            # Store the transformed data
            examples[output_column][ind] = transformed_data
            
        return examples


class SplitToFrame:
    def __call__(
        self,
        examples: dict[str, list],
        frame_length: int,
        hop_length: int,
        signal_column: str = "signal",
        key_column: str = "key"
    ) -> dict[str, list]:
        """Split signal to multiple frames with a sliding window.

        Args:
            examples (dict[str, list]): Input examples with signal and key columns
            frame_length (int): Frame length
            hop_length (int): Hop length
            signal_column (str): Name of the signal column
            key_column (str): Name of the key column

        Returns:
            dict[str, list]: Updated examples with split frames
        """
        new_signal, new_keys = [], []
        
        # Get other columns
        other_columns = list(examples.keys())
        other_columns.remove(signal_column)
        other_columns.remove(key_column)
        other_data = {key: [] for key in other_columns}

        # Process each signal and key pair
        for data_ind, (signal, key) in enumerate(
            zip(examples[signal_column], examples[key_column])
        ):
            frames = librosa.util.frame(
                signal, 
                frame_length=frame_length, 
                hop_length=hop_length, 
                axis=0
            )
            
            # Loop over the frames
            for ind, frame in enumerate(frames):
                if len(frame) != frame_length:
                    continue
                    
                new_signal.append(frame)
                new_keys.append(key + f";{ind}")
                
                # Copy other column values
                for column in other_columns:
                    other_data[column].append(examples[column][data_ind])

        # Construct output dictionary
        new_examples = {column: other_data[column] for column in other_columns}
        new_examples[signal_column] = new_signal
        new_examples[key_column] = new_keys
        
        return new_examples
    

    


class NormalizeSample:
    def __call__(
        self,
        examples: dict[str, list],
        signal_column: str = "signal",
        output_column: str = "signal"
    ) -> dict[str, list]:
        """Normalize samples by removing mean and scaling by standard deviation.

        Args:
            examples (dict[str, list]): Input examples
            signal_column (str): Name of input signal column
            output_column (str): Name of output column (can be same as input)

        Returns:
            dict[str, list]: Updated examples with normalized signals
        """
        # Initialize output column if different from input
        if signal_column != output_column:
            examples[output_column] = [None] * len(examples[signal_column])
            
        for ind, signal in enumerate(examples[signal_column]):
            std = np.std(signal, axis=-1, keepdims=True)
            examples[output_column][ind] = (signal - np.mean(signal, axis=-1, keepdims=True)) / std
            
        return examples