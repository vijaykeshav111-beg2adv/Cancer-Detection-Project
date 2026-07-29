

import os
import sys
import mlflow
mlflow.set_experiment('Cancer Detection Experiment')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from datapreprocessing.preprocess import Data  
from sklearn.model_selection import train_test_split
import pandas as pd  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier


PATH=r"C:\Users\kesha\OneDrive\Desktop\CancerDetectionProject\data\Cancer_Data.csv"
df = pd.read_csv(PATH)
x = df.drop(['diagnosis', 'id','Unnamed: 32'], axis=1)
y = df['diagnosis']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

pre_processor=Data()
preprocessor_transformer=pre_processor.preprocess_data(x_train)
x_train=preprocessor_transformer.fit_transform(x_train)

from sklearn.linear_model import LogisticRegression
model= LogisticRegression()
model.fit(x_train,y_train)
x_test=preprocessor_transformer.transform(x_test)
y_pred= model.predict(x_test)
from  sklearn.metrics import accuracy_score,classification_report,confusion_matrix,precision_score
score=accuracy_score(y_test,y_pred)
print(score)

models={
         "Logistic Regression":LogisticRegression(max_iter=100),
         "Decision Tree":DecisionTreeClassifier(),
         "Random Forest":RandomForestClassifier(),
        #  "Naive Bayes":MultinomialNB()

}
for name,algo in models.items():


# from  here  the model  experiment   is starting
        with  mlflow.start_run(run_name=name):
            
            algo.fit(x_train,y_train)
            y_pred=algo.predict(x_test)
        # here we are  caculating     the  metrics   of the  model 
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score,recall_score
            score=accuracy_score(y_test, y_pred)
            
            precision=precision_score(y_test, y_pred, pos_label='M')
            recall=recall_score(y_test, y_pred, pos_label='M')
            f1_score=f1_score(y_test, y_pred, pos_label='M')
            # storing the model metrics
            mlflow.log_metric("accuracy", score)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1_score)

            # logging the model
            mlflow.sklearn.log_model(algo, name=name)
            #massage 
            print(f'{name} saved sucessfully')
