import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import deque
import random
import editdistance
from ctc_loss import CTCLoss

import logging

class SlimIPL(pl.LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        tokenizer,
        preprocessor,
        spec_augmentation,
        unlabeled_dataloader,
        cache_size = 1000,
        cache_update_prob = 0.2,
        lambda_ratio = 0.7,
        initial_dropout = 0.5,
        final_dropout = 0.1,
        learning_rate = 2e-4,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.spec_augmentation = spec_augmentation
        self.unlabeled_dataloader = unlabeled_dataloader
        
        self.cache_size = cache_size
        self.cache_update_prob = cache_update_prob
        self.cache = deque(maxlen=cache_size)
        
        self.lambda_ratio = lambda_ratio
        self.initial_dropout = initial_dropout
        self.final_dropout = final_dropout
        self.learning_rate = learning_rate
        
        self.train_step = 0
        self.val_step = 0
        self.cache_filled = False
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
            input_audio = input_audio.to(self.device)
            input_audio_length = input_audio_length.to(self.device)
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
        if not self.cache_filled:
            input_audio, input_audio_len, input_text, input_text_len = batch
            logits, enc_len, preds = self.forward(input_signal=input_audio, input_signal_length=input_audio_len)
            loss = self.ctc_loss(log_probs=logits, targets=input_text, input_lengths=enc_len, target_lengths=input_text_len)
            
            if len(self.cache) < self.cache_size:
                unlabeled_batch = next(iter(self.unlabeled_dataloader))
                pseudo_labels = self._generate_pseudo_labels(unlabeled_batch)
                self.cache.append(unlabeled_batch + pseudo_labels)
            else:
                self.cache_filled = True
                self._set_dropout(self.final_dropout)   
        else:
            in_audio_labeled, in_audio_len_labeled, in_text_labeled, in_text_len_labeled = batch
            logits_labeled, enc_len_labeled, _ = self.forward(input_signal=in_audio_labeled, input_signal_length=in_audio_len_labeled)
            supervised_loss = self.ctc_loss(log_probs=logits_labeled, targets=in_text_labeled,
                                             input_lengths=enc_len_labeled, target_lengths=in_text_len_labeled)
            
            cached_batch = random.choice(list(self.cache))
            in_audio_unlabeled, in_audio_len_unlabeled, in_text_pseudo, in_text_len_pseudo = cached_batch 
            
            if random.random() < self.cache_update_prob:
                new_unlabeled_batch = next(iter(self.unlabeled_dataloader))
                new_pseudo_labels = self._generate_pseudo_labels(new_unlabeled_batch)
                self.cache.append(new_unlabeled_batch + new_pseudo_labels)
                
            logits_unlabeled, enc_len_ulabeled, _ = self.forward(input_signal=in_audio_unlabeled, input_signal_length=in_audio_len_unlabeled)
            unsupervised_loss = self.ctc_loss(log_probs=logits_unlabeled, targets=in_text_pseudo,
                                             input_lengths=enc_len_ulabeled, target_lengths=in_text_len_pseudo)

            self.log('unsupervised_train_loss', unsupervised_loss, on_epoch=False, on_step=True)
            self.log('supervised_train_loss', supervised_loss, on_epoch=False, on_step=True)
            
            loss = supervised_loss + self.lambda_ratio * unsupervised_loss
            
        self.log('train_loss', loss, on_epoch=False, on_step=True)

        self.train_step += 1

        return loss

    def validation_step(self, batch, batch_idx):
        self.eval()
        with torch.no_grad():
            input_audio, input_audio_len, input_text, input_text_len = batch
            logits, enc_len, preds = self.forward(input_signal=input_audio, input_signal_length=input_audio_len)

        val_loss = self.ctc_loss(log_probs=logits, targets=input_text, input_lengths=enc_len, target_lengths=input_text_len)
        self.log("val_loss", val_loss, on_epoch=True, on_step=False)

        decode_preds = self._decode_batch(preds)
        decode_preds = self.tokenizer.ids_to_text([x.tolist() for x in decode_preds])
        gt = self.tokenizer.ids_to_text([x[x!=0].tolist() for x in input_text])
        wer = self._word_error_rate(decode_preds, gt)
        self.log("val_wer", wer, on_epoch=True, on_step=False)
        
        self.val_step += 1

        return val_loss
    
    def configure_optimizers(self):
        parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.AdamW(
        parameters,
        lr=2e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1e4, eta_min=2e-4)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        return [optimizer], [scheduler]

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