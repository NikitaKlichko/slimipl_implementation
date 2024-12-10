from pytorch_lightning.loggers import TensorBoardLogger
from copy import deepcopy
from torch.utils.data import DataLoader
from data_module import DataModule
from utils import SpeechDataset, create_model_from_checkpoint
from slimipl import SlimIPL

logger = TensorBoardLogger('logs', name='run_model')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_path = ""
val_path = ""
unlabeld_path = ""
model_path = ""
train_bs = 32
val_bs = 8

encoder, decoder, tokenizer, spec_augment, preprocessor = create_model_from_checkpoint(model_path, device)


train_ds = SpeechDataset(train_path, tokenizer, is_labeled=True)
val_ds = SpeechDataset(val_path, tokenizer, is_labeled=True)
data_module = DataModule([train_ds], [val_ds], batch_size=train_bs)

pseudo_ds = SpeechDataset(unlabeld_path, tokenizer, is_labeled=False)
pseudo_loader = DataLoader(pseudo_ds, batch_size=val_bs, shuffle=False, collate_fn=pseudo_ds._speech_collate_fn, num_workers=4)

my_model = SlimIPL(encoder, decoder, tokenizer, preprocessor, spec_augment, unlabeled_dataloader=pseudo_loader,
)

trainer = pl.Trainer(
    max_epochs=10,
    logger=logger,
    gpus=1,
    precision='bf16', 
    accumulate_grad_batches=4,
    deterministic=True,
    save_top_k=1, 
    enable_model_summary=True,
    limit_train_batches=0.1, 
    limit_val_batches=0.1,  
    check_val_every_n_epoch=1,
)

trainer.fit(my_model, data_module)