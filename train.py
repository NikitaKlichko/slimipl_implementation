import argparse
import yaml
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from data_module import DataModule
from utils import SpeechDataset, create_model_from_checkpoint
from slimipl import SlimIPL
import torch


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_optimizer(parameters, config):
    optimizer_config = config['optimizer']
    if optimizer_config['type'] == "AdamW":
        optimizer = torch.optim.AdamW(
            parameters,
            lr=optimizer_config['lr'],
            betas=tuple(optimizer_config['betas']),
            eps=optimizer_config['eps'],
            weight_decay=optimizer_config['weight_decay']
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_config['type']}")
    return optimizer


def create_scheduler(optimizer, config):
    scheduler_config = config['scheduler']
    if scheduler_config['type'] == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config['mode'],
            patience=scheduler_config['patience'],
            factor=scheduler_config['factor']
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_config['type']}")
    return scheduler


def main(config):

    logger = TensorBoardLogger('./loggs', name='slimpl_run') if config['trainer']['logger'] == "tensorboard" else None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    encoder, decoder, tokenizer, spec_augment, preprocessor = create_model_from_checkpoint(config['model_path'], device)

    # Prepare datasets for training
    train_ds = [SpeechDataset(train_path, tokenizer, is_labeled=True) for train_path in config['train_paths']]
    
    # Prepare datasets for validation
    val_ds = [SpeechDataset(val_path, tokenizer, is_labeled=True) for val_path in config['val_paths']]
    
    concat_probs = config.get('concat_probs', None)

    # Create the data module
    data_module = DataModule(
        train_ds, val_ds,
        batch_size=config['train_bs'],
        n_workers=config['n_works'],
        concat_sampling_probabilities=concat_probs
    )

    # Prepare pseudo-labeled dataset for training
    pseudo_ds = SpeechDataset(config['unlabeled_path'], tokenizer, is_labeled=False)
    pseudo_loader = DataLoader(pseudo_ds, batch_size=config['train_bs'], shuffle=True, collate_fn=pseudo_ds._speech_collate_fn, num_workers=config['n_works'])

    # Create optimizer and scheduler
    optimizer = create_optimizer(list(encoder.parameters()) + list(decoder.parameters()), config)
    scheduler = create_scheduler(optimizer, config)

    # Initialize the SlimIPL model with parameters
    my_model = SlimIPL(
        encoder, decoder, tokenizer, preprocessor, spec_augment,
        unlabeled_dataloader=pseudo_loader,
        cache_size=config['cache_size'],
        supervised_updates=config['supervised_updates'],
        n_l_updates=config['n_l_updates'],
        n_u_updates=config['n_u_updates'],
        cache_update_prob=config['cache_update_prob'],
        initial_dropout=config['initial_dropout'],
        final_dropout=config['final_dropout'],
        optimizer=optimizer,
        scheduler=scheduler,
    )

    # Set up model checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor=config['trainer']['callbacks'][0]['monitor'],
        dirpath=config['trainer']['callbacks'][0]['dirpath'],
        filename=config['trainer']['callbacks'][0]['filename'],
        save_last=config['trainer']['callbacks'][0]['save_last'],
        save_top_k=config['trainer']['callbacks'][0]['save_top_k'],
        mode=config['trainer']['callbacks'][0]['mode'],
        save_weights_only=config['trainer']['callbacks'][0]['save_weights_only']
    )

    # Set up PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=config['trainer']['max_epochs'],
        accumulate_grad_batches=config['trainer']['accumulate_grad_batches'],
        check_val_every_n_epoch=config['trainer']['check_val_every_n_epoch'],
        log_every_n_steps=config['trainer']['log_every_n_steps'],
        devices=config['trainer']['devices'],
        accelerator=config['trainer']['accelerator'],
        precision=config['trainer']['precision'],
        enable_model_summary=config['trainer']['enable_model_summary'],
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    # Train the model
    trainer.fit(my_model, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the SlimIPL model.")

    # Path to config file
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file.")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Start the training process
    main(config)
