"""Train script for fault classification CNN model. Handles model training, validation and testing 
with early stopping and model checkpointing."""

import os
from typing import List
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.callback import Callback
from src.config import config
from src.data_module import DataModule
from src.lightning_module import LightningModel
from src.model import CNNModel

def setup_callbacks() -> List[Callback]:
    """Initialize training callbacks for model checkpointing and early stopping."""
    checkpoint_dir = "model_checkpoints"
    if os.path.exists(checkpoint_dir):
        import shutil
        shutil.rmtree(checkpoint_dir)
        os.makedirs(checkpoint_dir)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="model_checkpoints",
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor=config["early_stop_monitor"],
        min_delta=config["early_stop_mindelta"],
        patience=config["early_stop_patience"],
        verbose=True,
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    return [checkpoint_callback, lr_monitor, early_stop_callback]

def main() -> None:
    """Main training loop. Sets up model, data, and trainer for fault classification."""
    torch.set_float32_matmul_precision("high")
    callbacks = setup_callbacks()

    # Initialize modules
    data_module = DataModule(config)
    base_model = CNNModel(config)
    model = LightningModel(base_model, config)

    trainer = Trainer(
        max_epochs=config["max_epoch"],
        callbacks=callbacks,
        gradient_clip_val=config["gradient_clip_val"],
        accelerator="cuda",  # Change to "cpu" for CPU training
    )

    try:
        trainer.fit(model, datamodule=data_module)

        # Test best model
        checkpoint_callback = next(cb for cb in callbacks if isinstance(cb, ModelCheckpoint))
        best_model_path = checkpoint_callback.best_model_path

        if best_model_path:
            print(f"Loading best model from {best_model_path}")
            model = LightningModel.load_from_checkpoint(
                best_model_path, model=base_model, config=config
            )
            trainer.test(model, datamodule=data_module)

    except KeyboardInterrupt:
        print("Training interrupted by user.")

if __name__ == "__main__":
    main()