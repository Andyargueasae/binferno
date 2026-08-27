from pathlib import Path
from pydantic import BaseModel, Field
import loguru

class BaseReducer(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    
    pca: PcaReducer
    umap: UmapReducer
    use_pca: bool = Field(False, description="Whether to apply PCA for dimension reduction.")
    use_umap: bool = Field(False, description="Whether to apply UMAP for dimension reduction.")
    use_pipeline: bool = Field(False, description="Whether to apply a pipeline of PCA and UMAP for dimension reduction.")
    default_dim: int = Field(32, description="Default dimension for dimension reduction.")

    def __call__(self, data, logger: loguru._logger.Logger = None, threads: int = 1):
        raise NotImplementedError("Subclasses must implement the __call__ method.")