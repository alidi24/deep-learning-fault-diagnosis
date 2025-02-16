
# Model and Training Configuration
config = {
    "frame_length": 8192,
    "hop_length": 2048,
    "model_input_size": 8192,
    "batch_size": 32,
    "random_seed": 42,
    "test_size": 0.2,
    "val_size": 0.2,

    # Data augmentation
    "is_aug": "True",

    # Model architecture
    "num_classes": [4],
    "dropout_conv": 0.3,
    "dropout_fcl": 0.5,
    "cl_layer_num": 4,
    "conv_kernel_sizes": [15, 11, 7, 5],
    "dilation_rates": [1, 1, 1, 1],
    "ch_nums": [3, 32, 64, 128, 256],
    "fully_connected_layer": [32, 16],
    "adaptive_pool_size": 1,
    "act_fn": "ELU",
    "act_fn_out": "Softmax",
    "batch_normalization": "True",

    # Training parameters
    "optimizer_name": "Adam",
    "loss_fn": "CategoricalCrossentropy",
    "max_epoch": 60,
    "lr": 0.001,
    "weight_decay": 0.01,
    "early_stop_monitor": "val_loss",
    "early_stop_mindelta": 0.0,
    "early_stop_patience": 20,
    "lr_monitor": "val_loss",
    "lr_decay_factor": 0.9,
    "lr_patience": 10,
    "gradient_clip_val": None,

    # Preprocessing pipeline
    "preprocess_pipe": [
        "PCA() + "
        "SplitToFrame()"
    ],
}
