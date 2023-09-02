import pandas as pd
from torch.utils.data import DataLoader, Dataset
from features.base_features import BaseFeature
from dataset.base_dataset import InMemPandasDataSet
from sklearn.preprocessing import LabelEncoder
import numpy as np


column_names = ['age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education', 'wage_per_hour', 'hs_college',
                    'marital_stat', 'major_ind_code', 'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
                    'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses', 'stock_dividends',
                    'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ',
                    'instance_weight', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                    'num_emp', 'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                    'own_or_self', 'vet_question', 'vet_benefits', 'weeks_worked', 'year', 'income_50k']


class CensusFeatureInfo(BaseFeature):
    def step_transform(self, batch_dfs):
        df = pd.concat(batch_dfs)
        for feat in self.sparse_feats:
            lbe = LabelEncoder()
            df[feat.name] = lbe.fit_transform(df[feat.name])
        df['income_50k'] = self.to_categorical((df.income_50k == ' 50000+.').astype(int), num_classes=2)
        df['marital_stat'] = self.to_categorical((df.marital_stat == ' Never married').astype(int), num_classes=2)
        return df[self.feature_name].values, df[self.label_name].values

    @staticmethod
    def to_categorical(y, num_classes=None, dtype='float32'):
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, num_classes), dtype=dtype)
        categorical[np.arange(n), y] = 1
        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)
        return categorical


features = CensusFeatureInfo.from_yaml('conf/census_feature_info.yaml')
print(features.dense_feats)
print(features.sparse_feats)
print(features.varlen_sparse_feats)
print(features.label_info)

df_train = pd.read_csv('data/census/census-income.data.gz',  delimiter=',', header=None, index_col=None, names=column_names)
df_valid = pd.read_csv('data/census/census-income.test.gz',  delimiter=',', header=None, index_col=None, names=column_names)

train_dataset = InMemPandasDataSet(df_train)
valid_dataset = InMemPandasDataSet(df_valid)


train_dataloader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True, collate_fn=features.step_transform)
valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=256, shuffle=False, collate_fn=features.step_transform)

for X, y in train_dataloader:
    print(X.shape, y.shape)
    print(y)







# label_columns = ['income_50k', 'marital_stat']
# categorical_columns = ['class_worker', 'det_ind_code', 'det_occ_code', 'education', 'hs_college', 'major_ind_code',
#                            'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member', 'unemp_reason',
#                            'full_or_part_emp', 'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat',
#                            'det_hh_summ', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
#                            'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
#                            'vet_question']
# dense_feats = []
# for c in column_names:
#     if c not in categorical_columns and c not in label_columns:
#         dense_feats.append(c)


# dense_fea_info = dict()
# for feat in dense_feats:
#     dense_fea_info[feat] = {'name': feat, 'dimension': 1, 'dtype': 'float32'}
# sparse_fea_info = dict()
# for feat in categorical_columns:
#     lbe = LabelEncoder()
#     df_train[feat] = lbe.fit_transform(df_train[feat])
#     sparse_fea_info[feat] = {'name': feat,
#                              'vocabulary_size': df_train[feat].nunique(),
#                              'embedding_dim': 4}
# label_info = {'income_50k': {'name': 'income_50k', 'task_type': 'regression'},
#               'marital_stat': {'name': 'marital_stat', 'task_type': 'binary-classification'}}
#
#
#
# feature_info = Basefeature.write_feature_info(dense_feats=dense_fea_info,
#                                               sparse_feats=sparse_fea_info,
#                                               varlen_sparse_feats=None,
#                                               label_info=label_info,
#                                               yaml_path='conf/census_feature_info.yaml')



