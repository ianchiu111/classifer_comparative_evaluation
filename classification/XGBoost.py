
"""
1. Test this class with `python -m classification.XGBoost` in root

2. XGBoost wants features to be one of these:
    - int
    - float
    - bool
    - category
"""
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from typing import Dict
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from data_cleaning import ColumnConfig, DataCleaning

class XGBoostModel:
    """
    1. XGBoost is a gradient-boosted decision tree algorithm used for supervised learning.
    2. It can be used for abpve:
        1. classification
        2. regression
        3. ranking
        4. time-series forecasting after fine-tuning
    """
    
    def __init__(self):
        self.classification_params = {
            "objective": "binary:logistic",
            "max_depth": 6,
            "learning_rate": 0.01,
            "subsample": 0.8,
            "colsample": 0.8,
            "n_estimators": 300,
            "enable_categorical": False
        }

    def _train_classification(
            self, 
            X_train: pd.DataFrame, 
            X_test: pd.DataFrame,
            y_train: pd.DataFrame,
            y_test: pd.DataFrame,
            config: Dict = None,
            trained: bool = True
    ):
        """
        Training an XGBoost classification model
        default trained = True: There's already a trained model, so don't need to train again.
        """

        if trained == True:
            pass 
        else: 
            # train classification model
            if config is None:
                config = self.classification_params
            
            # set model config and train with fit function
            model = xgb.XGBClassifier(**config)
            model = model.fit(X_train, y_train)

            results = model.predict(X_test)
            report = classification_report(y_test, results)
            print("report:", report)

            with open('classification/models/xgboost-classification.pickle', 'wb') as f:
                pickle.dump(model, f)

if __name__ == "__main__":

    # ==================== Initiate Class ====================
    xgbmodel = XGBoostModel()
    titianic_training_config = ColumnConfig(
        int_cols = [
            "Survived", "Pclass", "Age", "SibSp", "Parch"
        ],
        float_cols = [
            "Fare"
        ],
        category_cols = [
            "Sex", "Cabin", "Embarked"
        ]
    )
    titianic_testing_config = ColumnConfig(
        int_cols = [
            "Pclass", "Age", "SibSp", "Parch"
        ],
        float_cols = [
            "Fare"
        ],
        category_cols = [
            "Sex", "Cabin", "Embarked"
        ]
    )

    titianic_training_cleaner = DataCleaning(columns = titianic_training_config)
    titianic_testing_cleaner = DataCleaning(columns = titianic_testing_config)

    # ==================== Data Processing ====================
    train_df = pd.read_csv("titanic/train_cleaned.csv")
    cleaned_train_df = titianic_training_cleaner.clean_data(data = train_df)
    answers = pd.read_csv("titanic/gender_submission.csv")

    y = cleaned_train_df['Survived']
    X = cleaned_train_df.drop(columns=['Survived'])

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_test, X_valid, y_test, y_valid = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


    # ==================== Training Model ====================
    xgbmodel._train_classification(
        X_train, 
        X_test,
        y_train, 
        y_test, 
        trained = True
    )

    # ==================== Test Model ====================
    try: 
        with open('classification/models/xgboost-classification.pickle', 'rb') as f:
            model = pickle.load(f)

    except FileNotFoundError:
        print("Model not found. Please train the model first.")

    test_df = pd.read_csv("titanic/test.csv")
    cleaned_test_df = titianic_testing_cleaner.clean_data(data = test_df)

    predictions = model.predict(cleaned_test_df)

    results = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Predicted": predictions
    })

    comparison = results.merge(
        answers,
        on="PassengerId",
        how="inner"
    )

    # np.where(condition, value_if_true, value_if_false)
    comparison["check"] = np.where(
        comparison["Predicted"] == comparison["Survived"],
        "correct",
        "wrong"
    )

    counts = comparison["check"].value_counts()
    correct = counts["correct"]
    wrong = counts["wrong"]
    score = correct/(correct+wrong)

    print("Correct Rate:", score)