import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoConfig
import time
from pydantic import Field
import loguru

from .base import BaseFeatureBuilder


class DnaBertSFeatureBuilder(BaseFeatureBuilder):

    max_len: int = Field(
        768,
        description="Maximum sequence length for DNABERT-S. Sequences longer than this will be chunked.",
    )
    stride: int = Field(
        512,
        description="Stride for chunking sequences longer than max_len. Determines the overlap between chunks.",
    )

    def __call__(
        self,
        logger: loguru._logger.Logger,
        fragments: list = [],
    ) -> pd.DataFrame:
        start_time = time.time()
        model_path = "zhihan1996/DNABERT-S"
        max_len = self.max_len  # < 1000 bp recommended.
        stride = self.stride

        logger.info(f"Using model checkpoint: {model_path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # the configuration is now changed completely.
        config = AutoConfig.from_pretrained(
            "yuhtong/DNABERT-S-binferno",
            trust_remote_code=True,
        )
        config.use_flash_attn = False
        config.use_flash_attention = False
        config.flash_attn = False
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # load model.
        model = AutoModel.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
            device_map=None,
        ).to(device)
        model.eval()

        features = []
        names = []
        total = len(fragments)
        logger.info(f"Embedding sequences: {total} contigs to process")

        for idx, (name, seq) in enumerate(fragments, start=1):
            if idx == 1 or idx % 100 == 0 or idx == total:
                logger.info(f"Embedding progress: {idx}/{total} contigs")

            names.append(name)
            emb_mean = self.embed_sequence(
                seq=seq,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_len=max_len,
                stride=stride,
            )

            if emb_mean is None:
                continue

            features.append(emb_mean)

        # form a ndarray first.
        features = np.asarray(features)
        assert (
            self.n_components < features.shape[1]
        ), f"n_components ({self.n_components}) must be less than feature dimension ({features.shape[1]})."

        df = pd.DataFrame(features, index=names)
        end_time = time.time()
        logger.info(f"Feature extraction completed in {end_time - start_time:.2f} seconds.")
        
        return df

    def embed_sequence(self, seq, tokenizer, model, device, max_len=512, stride=256):
        chunk_embs = []

        for start in range(0, len(seq), stride):
            sub_seq = seq[start : start + max_len]

            if len(sub_seq) < 50:
                continue

            encoded = tokenizer(
                sub_seq,
                return_tensors="pt",
                truncation=True,
                max_length=max_len,
            )

            encoded = {k: v.to(device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = model(**encoded)[0]  # (1, chunk_len, hidden_dim)

            chunk_emb = outputs.mean(dim=1).squeeze(0)  # (hidden_dim,)
            chunk_embs.append(chunk_emb.cpu())

        if len(chunk_embs) == 0:
            return None

        return torch.stack(chunk_embs, dim=0).mean(dim=0).numpy()
