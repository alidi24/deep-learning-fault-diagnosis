
import librosa
import numpy as np
from sklearn.decomposition import PCA


class PCATransform:
    def __init__(
            self, 
            n_components: int = 2,
            input_columns: list[str] = None,
            output_column: str = "pca_features",
            ):
        """Initialize the PCA transformation.
        
        Args:
            n_components (int): Number of components for PCA
            input_columns (list[str]): List of column names to apply PCA on
            output_column (str): Name of the output column
        """
        self.n_components = n_components
        self.input_columns = input_columns
        self.output_column = output_column

    def __call__(
        self, 
        examples: dict[str, list],
    ) -> dict[str, list]:
        """Apply PCA transformation to the input examples.
        
        Args:
            examples (dict[str, list]): Input examples with the specified columns 
        Returns:
            dict[str, list]: Updated examples with PCA output
        """
        if self.input_columns is None:
            raise ValueError("input_columns must be specified")
        
        if len(self.input_columns) < 2:
            raise ValueError("At least two input columns must be specified")
            
        pca = PCA(n_components=self.n_components)
        
        # Initialize output column
        examples[self.output_column] = [None] * len(examples[self.input_columns[0]])
        
        # Process each sample
        for ind in range(len(examples[self.input_columns[0]])):
            # Get data from all input columns for this sample
            sample_data = [np.array(examples[col][ind]).squeeze() for col in self.input_columns]
            data = np.vstack(sample_data).T
            
            # Apply PCA transformation
            transformed_data = pca.fit_transform(data)
            transformed_data = transformed_data.T
            
            # Store the transformed data
            examples[self.output_column][ind] = transformed_data
            
        return examples


class SplitToFrame:
    def __init__(
            self, 
            frame_length: int, 
            hop_length: int,
            signal_column: str = "signal",
            key_column: str = "key"
    ):
        """Initialize the frame splitter.
        
        Args:
            frame_length (int): Length of each frame
            hop_length (int): Hop length for sliding window
            signal_column (str): Name of the signal column
            key_column (str): Name of the key column
        """
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.signal_column = signal_column
        self.key_column = key_column


    def __call__(
        self,
        examples: dict[str, list]
    ) -> dict[str, list]:
        """Split signal to multiple frames with a sliding window.

        Args:
            examples (dict[str, list]): Input examples with signal and key columns

        Returns:
            dict[str, list]: Updated examples with split frames
        """
        new_signal, new_keys = [], []
        
        # Get other columns
        other_columns = list(examples.keys())
        other_columns.remove(self.signal_column)
        other_columns.remove(self.key_column)
        other_data = {key: [] for key in other_columns}

        # Process each signal and key pair
        for data_ind, (signal, key) in enumerate(
            zip(examples[self.signal_column], examples[self.key_column])
        ):
            signal = np.array(signal)
            frames = librosa.util.frame(
                signal, 
                frame_length=self.frame_length, 
                hop_length=self.hop_length, 
                axis=1
            ).squeeze()

            
            # Loop over the frames
            for ind, frame in enumerate(frames):
                if len(frame) != self.frame_length:
                    continue
                    
                new_signal.append(frame)
                new_keys.append(key + f";{ind}")
                
                # Copy other column values
                for column in other_columns:
                    other_data[column].append(examples[column][data_ind])

        # Construct output dictionary
        new_examples = {column: other_data[column] for column in other_columns}
        new_examples[self.signal_column] = new_signal
        new_examples[self.key_column] = new_keys
        
        return new_examples
    

    


# class NormalizeSample:
#     def __call__(
#         self,
#         examples: dict[str, list],
#         signal_column: str = "signal",
#         output_column: str = "signal"
#     ) -> dict[str, list]:
#         """Normalize samples by removing mean and scaling by standard deviation.

#         Args:
#             examples (dict[str, list]): Input examples
#             signal_column (str): Name of input signal column
#             output_column (str): Name of output column (can be same as input)

#         Returns:
#             dict[str, list]: Updated examples with normalized signals
#         """
#         # Initialize output column if different from input
#         if signal_column != output_column:
#             examples[output_column] = [None] * len(examples[signal_column])
            
#         for ind, signal in enumerate(examples[signal_column]):
#             std = np.std(signal, axis=-1, keepdims=True)
#             examples[output_column][ind] = (signal - np.mean(signal, axis=-1, keepdims=True)) / std
            
#         return examples