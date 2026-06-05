from pathlib import Path
from pydantic import BaseModel
import loguru

class BaseClustering(BaseModel):
    def __call__(
        self,
        embeddings: pd.DataFrame,
        seed_file: Path,
        contigs: Path,
        output_path: Path,
        contig_len: int = 1000,
        cluster_num: int = None,
        threads: int = 1,
        logger: loguru._logger.Logger = None,
    ) -> None:
        raise NotImplementedError("Subclasses must implement __call__ method.")

