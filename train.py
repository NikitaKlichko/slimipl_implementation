import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from data_module import DataModule
from utils import SpeechDataset, create_model_from_checkpoint
from slimipl import SlimIPL
import torch

logger = TensorBoardLogger('./loggs', name='slimpl_run')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_path = "..."
val_path = "..."
unlabeld_path = "..."
model_path = "..."

train_bs = 32
n_works = 4

encoder, decoder, tokenizer, spec_augment, preprocessor = create_model_from_checkpoint(model_path, device)

train_ds = SpeechDataset(train_path, tokenizer, is_labeled=True)
val_ds = SpeechDataset(val_path, tokenizer, is_labeled=True)
data_module = DataModule([train_ds], [val_ds], batch_size=train_bs, n_workers=n_works)

pseudo_ds = SpeechDataset(unlabeld_path, tokenizer, is_labeled=False)
pseudo_loader = DataLoader(pseudo_ds, batch_size=train_bs, shuffle=False, collate_fn=pseudo_ds._speech_collate_fn, num_workers=n_works)

my_model = SlimIPL(encoder, decoder, tokenizer, preprocessor, spec_augment, 
                    unlabeled_dataloader = pseudo_loader,
                    cache_size = int(len(train_ds) // train_bs * 0.1),
                    cache_update_prob = 0.2,
                    lambda_ratio = 0.7,
                    initial_dropout = 0.3,
                    final_dropout = 0.1,
                    learning_rate = 2e-4,
)

checkpoint_callback = ModelCheckpoint(
    monitor="val_wer",  
    dirpath="./ckpts/",  
    filename="best-checkpoint-{epoch:02d}-{val_wer:.3f}",  
    save_top_k=1, 
    mode="min", 
    save_weights_only=True 
)

trainer = pl.Trainer(
    max_epochs=10,
    logger=logger,
    callbacks=[checkpoint_callback],
    devices=1,
    accelerator="gpu",
    precision="bf16", 
    accumulate_grad_batches=4,
    enable_model_summary=True,
    # limit_train_batches=0.1, 
    # limit_val_batches=0.1,  
    check_val_every_n_epoch=1,
    log_every_n_steps=5,
)

trainer.fit(my_model, data_module)