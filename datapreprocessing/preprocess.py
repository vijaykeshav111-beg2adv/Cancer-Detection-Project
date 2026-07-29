from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import  Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import pandas as pd
class Data:
    # read the dataset
    @classmethod
    def preprocess_data(self,X):
       
    # seperating  the categorical and numerical columns
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
       # creating pipeline for categorical and numerical columns
        num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
                        ])


        cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
                    ])


        preprocessor = ColumnTransformer( transformers=[
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
        ])
        
        return preprocessor
        
    # missing values
    # scaling
    # encoding
    