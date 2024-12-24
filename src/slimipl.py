import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import deque
import random
import editdistance
from ctc_loss import CTCLoss
from utils import get_stdout_logger
import pickle

logging = get_stdout_logger("slimplog", "INFO")

class SlimIPL(pl.LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        tokenizer,
        preprocessor,
        spec_augmentation,
        unlabeled_dataloader,
        optimizer, 
        scheduler=None,
        supervised_updates=10000,
        n_l_updates = 1,
        n_u_updates = 4,
        cache_size = 1000,
        cache_update_prob = 0.1,
        initial_dropout = 0.5,
        final_dropout = 0.1,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.spec_augmentation = spec_augmentation
        self.unlabeled_dataloader = unlabeled_dataloader
        self.supervised_updates = supervised_updates
        
        self.cache_size = cache_size
        self.cache_update_prob = cache_update_prob
        self.n_l_updates = n_l_updates
        self.n_u_updates = n_u_updates
        self.cache = set()
        
        self.initial_dropout = initial_dropout
        self.final_dropout = final_dropout
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.train_step = 0
        self.val_step = 0
        self.cache_filled = False
        self.cntr_n_l = 0
        self.cntr_n_u = 0
        self.blank_id = self.decoder.num_classes_with_blank-1

        self.ctc_loss = CTCLoss(num_classes=self.blank_id, 
                                    reduction='mean_batch', 
                                    zero_infinity=True)
        
        self._set_dropout(initial_dropout)

    def _set_dropout(self, p):
        for m in self.encoder.modules():
            if isinstance(m, nn.Dropout):
                m.p = p

    def forward(self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None):
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )
        input_signal = input_signal.to(self.device)
        input_signal_length = input_signal_length.to(self.device)

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoder_output = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        encoded = encoder_output[0]
        encoded_len = encoder_output[1]
        log_probs = self.decoder(encoder_output=encoded)
        greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)

        return (
            log_probs,
            encoded_len,
            greedy_predictions,
        )

    def _generate_pseudo_labels(self, batch):
        self.eval()
        with torch.no_grad():
            input_audio, input_audio_length = batch
            logits, enc_len, preds = self.forward(input_signal=input_audio, input_signal_length=input_audio_length)
            pseudo_labels = self._form_labels_batch(preds)
        self.train()
        return pseudo_labels

    def _pad_sequences(self, sequences, pad_value=0):
        lengths = [len(seq) for seq in sequences]
        max_length = max(lengths)
        padded_sequences = []
        for seq in sequences:
            pad_size = max_length - len(seq)
            padded_seq = torch.nn.functional.pad(seq, (0, pad_size), value=pad_value)
            padded_sequences.append(padded_seq)
        return torch.stack(padded_sequences), torch.tensor(lengths).unsqueeze(1)
    
    def _clean_ids(self, predictions):
        clean_ids = []
        for sample in predictions:
            tokens_no_blank = sample[sample != self.blank_id]
            if tokens_no_blank.numel() == 0:
                clean_ids.append(tokens_no_blank)
                continue
            different_from_next = torch.cat([
                tokens_no_blank[1:] != tokens_no_blank[:-1],
                torch.tensor([True], device=tokens_no_blank.device)
            ])
            decoded_tokens = tokens_no_blank[different_from_next]
            clean_ids.append(decoded_tokens)
        return clean_ids

    def _decode_batch(self, predictions):
        return self._clean_ids(predictions)

    def _form_labels_batch(self, predictions):
        clean_ids = self._decode_batch(predictions)
        return self._pad_sequences(clean_ids, pad_value=0)

    def training_step(self, batch, batch_idx):
        in_audio_labeled, in_audio_len_labeled, in_text_labeled, in_text_len_labeled = batch

        if self.train_step < self.supervised_updates:
            logits, enc_len, preds = self.forward(input_signal=in_audio_labeled, input_signal_length=in_audio_len_labeled)
            supervised_loss = self.ctc_loss(log_probs=logits, targets=in_text_labeled, input_lengths=enc_len, target_lengths=in_text_len_labeled)

            self.train_step += 1

            self.log('supervised_train_loss', supervised_loss, on_epoch=False, on_step=True)

            return supervised_loss
            
        elif not self.cache_filled:
            if len(self.cache) < self.cache_size:
                unlabeled_batch = next(iter(self.unlabeled_dataloader))
                pseudo_labels = self._generate_pseudo_labels(unlabeled_batch)
                self.cache.add(unlabeled_batch + pseudo_labels)

                logits_labeled, enc_len_labeled, labeled_preds = self.forward(input_signal=in_audio_labeled, input_signal_length=in_audio_len_labeled)
                supervised_loss = self.ctc_loss(log_probs=logits_labeled, targets=in_text_labeled,
                                                input_lengths=enc_len_labeled, target_lengths=in_text_len_labeled)

                self.log('supervised_train_loss', supervised_loss, on_epoch=False, on_step=True)

                return supervised_loss
            else:
                self.cache_filled = True
                logging.info(f"Cache of size {self.cache_size} is full")
            
                self._set_dropout(self.final_dropout)
                
                return None   
        else:
            if self.cntr_n_l < self.n_l_updates:
                logits_labeled, enc_len_labeled, labeled_preds = self.forward(input_signal=in_audio_labeled, input_signal_length=in_audio_len_labeled)
                supervised_loss = self.ctc_loss(log_probs=logits_labeled, targets=in_text_labeled,
                                                input_lengths=enc_len_labeled, target_lengths=in_text_len_labeled)

                self.log('supervised_train_loss', supervised_loss, on_epoch=False, on_step=True)
                
                supervised_wer = self.compute_wer(labeled_preds, in_text_labeled, False)

                self.log('supervised_train_wer', supervised_wer, on_epoch=False, on_step=True)

                self.cntr_n_l += 1
                logging.info(f"counter labeled: {self.cntr_n_l}")

                return supervised_loss
            else:
                cached_batch = random.choice(list(self.cache))

                in_audio_unlabeled, in_audio_len_unlabeled, in_text_pseudo, in_text_len_pseudo = cached_batch 
                
                if random.random() < self.cache_update_prob:
                    self.cache.remove(cached_batch)
                    new_unlabeled_batch = next(iter(self.unlabeled_dataloader))
                    new_pseudo_labels = self._generate_pseudo_labels(new_unlabeled_batch)
                    self.cache.add(new_unlabeled_batch + new_pseudo_labels)
                    
                logits_unlabeled, enc_len_ulabeled, unlabeled_preds = self.forward(input_signal=in_audio_unlabeled, input_signal_length=in_audio_len_unlabeled)
                unsupervised_loss = self.ctc_loss(log_probs=logits_unlabeled, targets=in_text_pseudo,
                                                input_lengths=enc_len_ulabeled, target_lengths=in_text_len_pseudo)

                self.log('unsupervised_train_loss', unsupervised_loss, on_epoch=False, on_step=True)

                unsupervised_wer = self.compute_wer(unlabeled_preds, in_text_pseudo, False)

                self.log('unsupervised_train_wer', unsupervised_wer, on_epoch=False, on_step=True)

                self.cntr_n_u += 1

                if (self.cntr_n_u == self.n_u_updates) and (self.cntr_n_l == self.n_l_updates):
                    self.cntr_n_u = 0
                    self.cntr_n_l = 0

                return unsupervised_loss

    def compute_wer(self, predicitons, target_text, log=True):
        decode_preds = self._decode_batch(predicitons)
        decode_preds = self.tokenizer.ids_to_text([x.tolist() for x in decode_preds])

        gt = self.tokenizer.ids_to_text([x[x!=0].tolist() for x in target_text])
        wer = self._word_error_rate(decode_preds, gt)
        if log:   
            logging.info(f"\n")
            logging.info(f"reference:{gt[0]}")
            logging.info(f"predicted:{decode_preds[0]}")

        return wer

    def validation_step(self, batch, batch_idx):
        self.eval()
        with torch.no_grad():
            input_audio, input_audio_len, input_text, input_text_len = batch
            logits, enc_len, preds = self.forward(input_signal=input_audio, input_signal_length=input_audio_len)

        val_loss = self.ctc_loss(log_probs=logits, targets=input_text, input_lengths=enc_len, target_lengths=input_text_len)
   
        self.log("val_loss", val_loss, on_epoch=True, on_step=False)
        val_wer = self.compute_wer(preds, input_text)

        self.val_step += 1

        self.log("val_wer", val_wer, on_epoch=True, on_step=False)

        return val_loss
    
    def configure_optimizers(self):
        if self.scheduler:
            return [self.optimizer], [{"scheduler": self.scheduler, "monitor": "val_wer", "interval": "epoch", "frequency": 1}]
        else:
            return [self.optimizer]

    def _word_error_rate(self, hypotheses, references, use_cer=False):
        scores = 0
        words = 0
        if len(hypotheses) != len(references):
            raise ValueError(
                "In word error rate calculation, hypotheses and reference"
                " lists must have the same number of elements. But I got:"
                "{0} and {1} correspondingly".format(len(hypotheses), len(references))
            )
        for h, r in zip(hypotheses, references):
            if use_cer:
                h_list = list(h)
                r_list = list(r)
            else:
                h_list = h.split()
                r_list = r.split()
            words += len(r_list)
            scores += editdistance.eval(h_list, r_list)
        if words != 0:
            wer = 1.0 * scores / words
        else:
            wer = float('inf')
        return wer