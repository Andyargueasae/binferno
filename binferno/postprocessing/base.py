
from pathlib import Path
from pydantic import BaseModel
import loguru

class BasePostprocessor(BaseModel):
    def __call__(
        self,
        contigs: Path,
        output_path: Path,
        seed_num: int = 0,
        ignore_kmeans_res: bool = False,
        num_threads: int = 5,
        logger: loguru._logger.Logger = None,
        res_name: str = None,
        plastid: bool = False,
    ) -> None:
        raise NotImplementedError(
            "BasePostprocessor is an abstract class. Please implement the __call__ method."
        )
