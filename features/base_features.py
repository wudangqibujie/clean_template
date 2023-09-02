from dataclasses import dataclass
from typing import List, Optional
import yaml


@dataclass
class SparseFeat:
    name: str
    vocabulary_size: int
    embedding_dim: int
    use_hash: bool = False
    dtype: str = "int32"
    embedding_name: Optional[str] = None

    def __post_init__(self):
        self.embedding_name = self.name + "_embedding" if self.embedding_name is None else self.embedding_name


@dataclass
class DenseFeat:
    name: str
    dimension: int
    dtype: str = "float32"


@dataclass
class VarLenSparseFeat:
    name: str
    vocabulary_size: int
    embedding_dim: int
    maxlen: int
    combiner: str = "mean"
    use_hash: bool = False
    dtype: str = "float32"
    embedding_name: Optional[str] = None


@dataclass
class LabelInfo:
    name: str
    task_type: str = "binary-classification"


class BaseFeature:
    def __init__(self, feature_info):
        self.feature_info = feature_info

    def update_feature_info(self, feat_name, feature_info):
        if isinstance(feature_info, DenseFeat):
            self.feature_info['dense_feats'][feat_name] = feature_info
        elif isinstance(feature_info, SparseFeat):
            self.feature_info['sparse_feats'][feat_name] = feature_info
        elif isinstance(feature_info, VarLenSparseFeat):
            self.feature_info['varlen_sparse_feats'][feat_name] = feature_info
        elif isinstance(feature_info, LabelInfo):
            self.feature_info['label_info'] = feature_info
        else:
            raise ValueError("feature_info should be one of [DenseFeat, SparseFeat, VarLenSparseFeat, LabelInfo]")

    def step_filter(self, df):
        return df

    def step_transform(self, df):
        return df

    @property
    def dense_feats(self):
        return [DenseFeat(**feat) for feat in self.feature_info['dense_feats'].values()] if self.feature_info['dense_feats'] else []

    @property
    def sparse_feats(self):
        return [SparseFeat(**feat) for feat in self.feature_info['sparse_feats'].values()] if self.feature_info['sparse_feats'] else []

    @property
    def label_name(self):
        return [label.name for label in self.label_info]

    @property
    def feature_name(self):
        return [feat.name for feat in self.sparse_feats + self.dense_feats + self.varlen_sparse_feats]

    @property
    def varlen_sparse_feats(self):
        return [VarLenSparseFeat(**feat) for feat in self.feature_info['varlen_sparse_feats'].values()] if self.feature_info['varlen_sparse_feats'] else []

    @property
    def label_info(self):
        return [LabelInfo(**label_info) for label_info in self.feature_info['label_info'].values()] if self.feature_info['label_info'] else []

    @classmethod
    def from_yaml(cls, yaml_file):
        with open(yaml_file, 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
            return cls(data)

    @staticmethod
    def write_feature_info(dense_feats, sparse_feats, varlen_sparse_feats, label_info, yaml_path):
        features_info = {}
        features_info['dense_feats'] = dense_feats
        features_info['sparse_feats'] = sparse_feats
        features_info['varlen_sparse_feats'] = varlen_sparse_feats
        features_info['label_info'] = label_info
        with open(yaml_path, 'w') as file:
            yaml.dump(features_info, file)



