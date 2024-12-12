import torch
import json
from torch.utils.data import Dataset
import nemo.collections.asr as nemo_asr
import torchaudio
from copy import deepcopy
import logging
import sys


class SpeechDataset(Dataset):
    def __init__(self, data_path, tokenizer, use_start_end=False, is_labeled=True):
        self.is_labeled = is_labeled
        self.tokenizer = tokenizer
        self.use_start_end = use_start_end

        self.data = self.read_manifest(data_path)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        audio = self.data[idx]['audio_filepath']
        audio = self.prepare_audio(audio)
        audio_len = torch.tensor(audio.size(0))
        if self.is_labeled:
            txt = self.data[idx]['text']
            encoded_txt = self.tokenize_txt(txt, self.tokenizer, self.use_start_end)
            encoded_txt = torch.tensor(encoded_txt, dtype=torch.long)
            encoded_len = torch.tensor([encoded_txt.size(0)])
            return (audio,
                    audio_len,
                    encoded_txt,
                    encoded_len,
                    )
        return (audio, audio_len, None, None)

    def tokenize_txt(self, txt, tokenizer, use_start_end=False):
        tokenized_txt = tokenizer.text_to_ids(txt)
        if self.use_start_end:
            bos_id = tokenizer.bos_id
            eos_id = tokenizer.eos_id
            if hasattr(tokenizer, 'bos_id') and bos_id > 0:
                tokenized_txt = [bos_id] + tokenized_txt
            if hasattr(tokenizer, 'eos_id') and eos_id > 0:
                tokenized_txt = tokenized_txt + [eos_id]
        return tokenized_txt

    def read_manifest(self, data_path):
        with open(data_path, 'r') as f:
            data = f.readlines()

        data_list = []
        for i in data:
            data_list.append(json.loads(i))

        return data_list

    def prepare_audio(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)
        return waveform[0]

    def _speech_collate_fn(self, batch):
        pad_id = self.tokenizer.pad_id if self.tokenizer.pad_id > 0 else 0
        
        packed_batch = list(zip(*batch))
        if len(packed_batch) == 5:
            _, audio_lengths, _, tokens_lengths, sample_ids = packed_batch
        elif len(packed_batch) == 4:
            sample_ids = None
            _, audio_lengths, _, tokens_lengths = packed_batch
        else:
            raise ValueError("Expects 4 or 5 tensors in the batch!")
        max_audio_len = 0
        has_audio = audio_lengths[0] is not None
        if has_audio:
            max_audio_len = max(audio_lengths).item()
        has_tokens = tokens_lengths[0] is not None
        if has_tokens:
            max_tokens_len = max(tokens_lengths).item()

        audio_signal, tokens = [], []
        for b in batch:
            if len(b) == 5:
                sig, sig_len, tokens_i, tokens_i_len, _ = b
            else:
                sig, sig_len, tokens_i, tokens_i_len = b
            if has_audio:
                sig_len = sig_len.item()
                if sig_len < max_audio_len:
                    pad = (0, max_audio_len - sig_len)
                    sig = torch.nn.functional.pad(sig, pad)
                audio_signal.append(sig)

            if has_tokens:
                tokens_i_len = tokens_i_len.item()
                if tokens_i_len < max_tokens_len:
                    pad = (0, max_tokens_len - tokens_i_len)
                    tokens_i = torch.nn.functional.pad(tokens_i, pad, value=pad_id)
                tokens.append(tokens_i)

        if has_audio:
            audio_signal = torch.stack(audio_signal)
            audio_lengths = torch.stack(audio_lengths)
        else:
            audio_signal, audio_lengths = None, None
        if has_tokens:
            tokens = torch.stack(tokens)
            tokens_lengths = torch.stack(tokens_lengths)
        else:
            tokens = None
            tokens_lengths = None
        if sample_ids is None:
            if self.is_labeled:
                return audio_signal, audio_lengths, tokens, tokens_lengths
            else:
                return audio_signal, audio_lengths
        else:
            sample_ids = torch.tensor(sample_ids, dtype=torch.int32)
            if self.is_labeled:
                return audio_signal, audio_lengths, tokens, tokens_lengths, sample_ids
            else:
                return audio_signal, audio_lengths, sample_ids


def create_model_from_checkpoint(model_path, device):
    ctc_conformer = nemo_asr.models.EncDecCTCModelBPE.restore_from(model_path)
    encoder = deepcopy(ctc_conformer.encoder).to(device)
    decoder = deepcopy(ctc_conformer.decoder).to(device)
    tokenizer = deepcopy(ctc_conformer.tokenizer)
    spec_augment = deepcopy(ctc_conformer.spec_augmentation).to(device)
    preprocessor = deepcopy(ctc_conformer.preprocessor).to(device)
    del ctc_conformer
    return encoder, decoder, tokenizer, spec_augment, preprocessor

def get_stdout_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger