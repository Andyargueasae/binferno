from pydantic import BaseModel
import pandas as pd

class BaseFeatures(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    def sort_features_by_method(
        self,
        features_df: pd.DataFrame,
        simclr: bool = False,
        contig_ids: list = None,
    ) -> pd.DataFrame:
        if not contig_ids:
            raise ValueError(
                "contig_ids is required for sorting features. Please provide the list of contig names in the "
                "order they were processed during augmentation."
            )
        if not simclr:
            # sort the features_df by index.
            features_df = features_df.sort_index()
            return features_df
        else:
            # for simclr, group by view, stack the features of same view together, then concatenate the features.
            features_df["view"] = (
                features_df.index.str.rsplit(".", n=1).str[-1].astype(str)
            )
            # replace "original" as "0".
            # features_df["view"] = features_df["view"].replace("original", "0")
            features_df["view"] = features_df["view"].astype(int)

            features_df = features_df.sort_values(by=["view"], ascending=True)
            for i, view in enumerate(features_df["view"].unique()):
                view_df = features_df[features_df["view"] == view]
                # reindex view_df to make sure the order of features is the same across views.
                view_df.index = [ind.rsplit(".", 1)[0] for ind in view_df.index]
                view_df = view_df.reindex(
                    contig_ids
                )  # reindexing to ensure the same order of contigs across views
                view_df.index = [
                    f"{idx}.{view}" for idx in view_df.index
                ]  # restore the original index format after reindexing
                if i == 0:
                    sorted_features_df = view_df
                else:
                    sorted_features_df = pd.concat(
                        [sorted_features_df, view_df], axis=0
                    )
            # for each view, make sure to sort the features by index.
            features_df = sorted_features_df.drop(columns=["view"])
            return features_df

    # def multi_sample_feature_sort(self,...):
    #     pass 