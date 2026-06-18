from pathlib import Path

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoConfig
import pyfastx
import rich.progress as progress
import time
from pydantic import Field

from .base import BaseFeatureBuilder

class DnaBertSFeatureBuilder(BaseFeatureBuilder):
    pca: bool = Field(False, description="Whether to apply PCA to reduce dimensionality of features.")
    n_components: int = Field(128, description="Number of PCA components to retain if PCA is applied.")

    def __call__(self,
                fragments: list = [],
        ) -> pd.DataFrame:
        start_time = time.time()
        model_path = "zhihan1996/DNABERT-S"
        max_len = 768 # < 1000 bp recommended.
        stride = 512

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seqs = [(name, str(seq)) for name, seq in fragments if len(seq) > 1000]

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
        for name, seq in progress.track(seqs, description="Embedding sequences"):
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
        assert self.n_components < features.shape[1], f"n_components ({self.n_components}) must be less than feature dimension ({features.shape[1]})."
        
        n_components = min(self.n_components, features.shape[0], features.shape[1])

        if n_components < 2:
            # too few contigs for meaningful PCA
            pass
        else:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components)
            features = pca.fit_transform(features)

        if features.shape[1] < self.n_components:
            features = np.pad(
                features,
                ((0, 0), (0, self.n_components - features.shape[1])),
                mode="constant",
            )

        df = pd.DataFrame(features, index=names)
        end_time = time.time()
        print(f"Feature extraction completed in {end_time - start_time:.2f} seconds.")
        return df

    def embed_sequence(self, seq, tokenizer, model, device, max_len=512, stride=256):
        chunk_embs = []

        for start in progress.track(range(0, len(seq), stride), description="Embedding chunks"):
            sub_seq = seq[start:start + max_len]

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