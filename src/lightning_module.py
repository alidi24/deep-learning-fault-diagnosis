import os
from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam, SGD, AdamW, lr_scheduler
from torchinfo import summary
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassConfusionMatrix
import seaborn as sns
import matplotlib.pyplot as plt


class LightningModel(pl.LightningModule):
    """Lightning Module for multi-class fault classification."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]) -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.setup_metrics()

    def setup_metrics(self) -> None:
        """Setup model metrics for training, validation and testing."""
        self.loss_fn = nn.CrossEntropyLoss()
        num_classes = self.config["num_classes"]

        # Training metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)

        # Validation metrics
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_confusion = MulticlassConfusionMatrix(num_classes=num_classes)

        # Test metrics
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_confusion = MulticlassConfusionMatrix(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        return self.model(x)



    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = self.train_acc(y_hat, y)
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = self.val_acc(y_hat, y)
        self.val_confusion.update(y_hat, y)
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step."""
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = self.test_acc(y_hat, y)
        self.test_confusion.update(y_hat, y)
        
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        
        return {"test_loss": loss, "test_acc": acc}

    def on_test_epoch_end(self) -> None:
        """Plot confusion matrix at the end of test epoch."""
        conf_matrix = self.test_confusion.compute().cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        
        # Save the plot
        save_path = os.path.join(self.trainer.default_root_dir, 'confusion_matrix.png')
        plt.savefig(save_path)
        plt.close()
        
        print(f"\nConfusion matrix saved to: {save_path}")
        
        # Log final test accuracy
        final_acc = self.test_acc.compute()
        print(f"Final Test Accuracy: {final_acc:.4f}")

    def configure_optimizers(self) -> Dict[str, Any]:

        optimizer_name = self.config.get("optimizer_name", "adam").upper()

        if optimizer_name == "ADAM":
            optimizer = Adam(self.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"])
        elif optimizer_name == "ADAMW":
            optimizer = AdamW(self.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"])
        elif optimizer_name == "SGD":
            optimizer = SGD(
                self.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"], momentum=0.9
            )
        else:  # default to Adam
            print(f"Warning: Unsupported optimizer '{self.config.get('optimizer_name')}'. Using AdamW as default.")
            optimizer = AdamW(self.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"])

        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.config["lr_decay_factor"],
            patience=self.config["lr_patience"],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.config["lr_monitor"],
                "frequency": 1,
                "strict": True,
            },
        }