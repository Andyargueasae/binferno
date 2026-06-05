
from pathlib import Path
from pydantic import BaseModel

class BaseWriter(BaseModel):
    def __call__(
        self,
        contigs: Path,
        cluster_res_dir: Path,
        output_path: Path,
        plastid: bool = False,
        minbin_size: int = 200000,
    ) -> None:
        raise NotImplementedError(
            "BaseWriter is an abstract class. Please implement the __call__ method."
        )
