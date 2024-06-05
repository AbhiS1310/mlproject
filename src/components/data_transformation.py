import sys
import os
import pandas as pd 
import numpy as np 

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_obj(self):
        try:
            num_features = ["writing_score", "reading_score"]
            cat_features = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scalar',StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scalar',StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ('num_pipelines',num_pipeline,num_features),
                    ('cat_pipelines',cat_pipeline,cat_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    def intiate_data_transformation(self):
        try:
            train_df = pd.read_csv('artifacts/train.csv')
            test_df = pd.read_csv('artifacts/test.csv')

            target_feature = 'math_score'

            input_feature_train_df = train_df.drop(columns=[target_feature],axis=1)
            target_feature_train_df = train_df[target_feature]

            input_feature_test_df = test_df.drop(columns=[target_feature],axis=1)
            target_feature_test_df = test_df[target_feature]

            preprocessing_obj = self.get_data_transformer_obj()

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_obj(
                self.data_transformation_config.preprocessor_file_path,
                preprocessing_obj
            )

            logging.info("Data Transformation is complete")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_file_path

            )

        except Exception as e:
            raise CustomException(e,sys)