from zenml import step
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from typing import Tuple
from typing_extensions import Annotated

class TFTLightningModule(pl.LightningModule):
    def __init__(self, model: TemporalFusionTransformer, learning_rate: float, reduce_on_plateau_patience: int):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=['model'])
        # Access learning_rate and reduce_on_plateau_patience via self.hparams after save_hyperparameters

    def _ensure_batch_on_device(self, batch, target_device):
        x_orig, y_orig = batch
        # Ensure x_orig (dict of tensors) is on the target_device
        x_on_device = {
            k: v.to(target_device) if isinstance(v, torch.Tensor) and v.device != target_device else v
            for k, v in x_orig.items()
        }
        
        # Ensure y_orig (tuple of tensors) is on the target_device
        if isinstance(y_orig, tuple):
            y_on_device = tuple(
                t.to(target_device) if isinstance(t, torch.Tensor) and t.device != target_device else t
                for t in y_orig
            )
        elif isinstance(y_orig, torch.Tensor): # Should be a tuple based on TimeSeriesDataSet
            y_on_device = y_orig.to(target_device) if y_orig.device != target_device else y_orig
        else:
            y_on_device = y_orig # Should not happen if y contains tensors as expected
            
        return x_on_device, y_on_device

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        # Explicitly ensure the batch is on the correct device
        batch = self._ensure_batch_on_device(batch, self.device)
        
        # self.model is the TemporalFusionTransformer instance
        # Its training_step is inherited from BaseModel and expects the batch (x_dict, y_tuple)
        output_dict = self.model.training_step(batch, batch_idx)
        loss = output_dict["loss"]
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        # Explicitly ensure the batch is on the correct device
        batch = self._ensure_batch_on_device(batch, self.device)
        
        output_dict = self.model.validation_step(batch, batch_idx)
        loss = output_dict["loss"]
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Explicitly ensure the batch is on the correct device
        batch = self._ensure_batch_on_device(batch, self.device)
        return self.model.predict_step(batch, batch_idx, dataloader_idx=dataloader_idx)

    def configure_optimizers(self):
        if hasattr(self.model, 'configure_optimizers') and callable(getattr(self.model, 'configure_optimizers')):
            try:
                # Attempt to use the wrapped model's own optimizer configuration
                opt_config = self.model.configure_optimizers()
                # Ensure it's in the format [optimizers], [schedulers] if not already
                if isinstance(opt_config, tuple) and len(opt_config) == 2:
                    return opt_config # Expected format [optimizers], [schedulers]
                elif isinstance(opt_config, torch.optim.Optimizer):
                    return [opt_config] # Only optimizer provided
                # Add more checks if TFT returns other formats
                return opt_config # Return as is if format is already correct
            except Exception as e:
                print(f"Could not use wrapped model's configure_optimizers due to: {e}. Falling back.")

        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate) # Use self.parameters() for the wrapper
        scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.1,
                patience=self.hparams.reduce_on_plateau_patience,
                min_lr=1e-6,
                verbose=True,
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [scheduler_config]

@step
def train_model(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    forecast_horizon: int = 7,
    hidden_size: int = 32,
    dropout: float = 0.2,
    learning_rate: float = 0.005,
    max_encoder_length: int = 14,
    min_encoder_length: int = 7,
    batch_size: int = 32,
    max_epochs: int = 30,
) -> Tuple[
    Annotated[pl.LightningModule, "model"],
    Annotated[TimeSeriesDataSet, "training_dataset_config"]
]:
    """
    Train a Temporal Fusion Transformer (TFT) model for retail forecasting.
    This version simplifies dataset creation and model initialization.
    """
    # Diagnostic prints
    print(f"Train data shape: {train_data.shape}")
    if not train_data.empty:
        print(f"Train time_idx range: {train_data['time_idx'].min()}-{train_data['time_idx'].max()}")
        series_counts = train_data.groupby('series_id')['time_idx'].count()
        print(f"Training series length - Min: {series_counts.min()}, Mean: {series_counts.mean():.1f}, Max: {series_counts.max()}")
    
    print(f"Val data shape: {val_data.shape}")
    if not val_data.empty:
        print(f"Val time_idx range: {val_data['time_idx'].min()}-{val_data['time_idx'].max()}")

    # Ensure min_encoder_length is not greater than max_encoder_length
    actual_min_encoder_length = min(min_encoder_length, max_encoder_length)
    print(f"Using max_encoder_length: {max_encoder_length}, actual_min_encoder_length: {actual_min_encoder_length}, forecast_horizon: {forecast_horizon}")

    # Define feature columns (ensure they exist in train_data)
    static_categoricals = [col for col in ["store_encoded", "item_encoded"] if col in train_data.columns]
    time_varying_known_categoricals = [col for col in ["day_of_week", "month"] if col in train_data.columns]
    time_varying_known_reals = [col for col in ["is_holiday", "is_promo", "is_weekend"] if col in train_data.columns]
    
    time_varying_unknown_reals = ["sales"] + [
        col for col in train_data.columns
        if col.startswith(("sales_lag_", "sales_rolling_mean_", "sales_rolling_std_"))
    ]
    
    categorical_encoder_cols = [col for col in static_categoricals + time_varying_known_categoricals if col in train_data.columns]

    # Create training TimeSeriesDataSet
    training_dataset = TimeSeriesDataSet(
        data=train_data,
        time_idx="time_idx",
        target="sales",
        group_ids=["series_id"],
        max_encoder_length=max_encoder_length,
        min_encoder_length=actual_min_encoder_length,
        max_prediction_length=forecast_horizon,
        static_categoricals=static_categoricals,
        static_reals=[], 
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(groups=["series_id"], transformation="softplus"),
        categorical_encoders={cat: NaNLabelEncoder(add_nan=True) for cat in categorical_encoder_cols},
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    # Create validation TimeSeriesDataSet from training_dataset
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        val_data,
        stop_randomization=True,
        allow_missing_timesteps=True,
        # predict_mode=True # Consider if sequence length issues persist in validation
    )

    # Create dataloaders
    train_dataloader = training_dataset.to_dataloader(
        train=True, batch_size=batch_size, num_workers=1, pin_memory=True
    )
    val_dataloader = validation_dataset.to_dataloader(
        train=False, batch_size=batch_size * 2, num_workers=1, pin_memory=True
    )

    # Initialize the Temporal Fusion Transformer model
    tft_model_instance = TemporalFusionTransformer.from_dataset(
        training_dataset, 
        learning_rate=learning_rate, # This LR is for TFT internal use if it configures its own optimizer
        hidden_size=hidden_size,
        attention_head_size=4, 
        dropout=dropout,
        hidden_continuous_size=hidden_size // 2, 
        loss=QuantileLoss(),
        optimizer="adam", # TFT will try to use Adam
        reduce_on_plateau_patience=5, # For TFT internal scheduler if used
    )
    print(f"Number of parameters in model: {tft_model_instance.size() / 1e3:.1f}k")

    # Wrap the TFT model instance in our LightningModule
    lightning_model = TFTLightningModule(
        model=tft_model_instance, 
        learning_rate=learning_rate, # Pass LR for our wrapper's optimizer config
        reduce_on_plateau_patience=5 # Pass patience for our wrapper's scheduler config
    )

    # Configure PyTorch Lightning Trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=True, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        # accelerator="gpu" if torch.cuda.is_available() else "cpu",
        # devices=1 if torch.cuda.is_available() else None,
        accelerator="cpu",
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback, lr_monitor],
        logger=True, 
        enable_model_summary=True,
    )

    # Train the model
    print("Training TFT model...")
    trainer.fit(
        lightning_model, # Pass the wrapped model
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    return lightning_model, training_dataset # Return the wrapped model
