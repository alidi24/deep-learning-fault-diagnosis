
# Model and Training Configuration
config = {
    "frame_length": 8192,
    "hop_length": 6144,
    "batch_size": 32,
    "random_seed": 42,
    "test_size": 0.2,
    "val_size": 0.1,

    # Data augmentation
    "is_aug": "True",

    # Model architecture
    "num_classes": 4,
    "dropout_conv": 0.2,
    "dropout_fcl": 0.4,
    "cl_layer_num": 3,
    "conv_kernel_sizes": [21, 3, 3],
    "ch_nums": [1, 16, 32, 64],
    "fully_connected_layer": [64, 32],
    "adaptive_pool_size": 32,
    "act_fn": "ELU",
    "act_fn_out": "Softmax",
    "batch_normalization": "True",

    # Training parameters
    "optimizer_name": "Adam",
    "loss_fn": "CategoricalCrossentropy",
    "max_epoch": 20,
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
