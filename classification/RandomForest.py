
"""
1. Test this class with `python -m classification.RandomForest` in root

2. Random Forest wants features to be one of these:
    - int
    - float
    - bool
3. Because Random Forest cannot accept string or category columns, so we try dummy instead.
"""
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from typing import Dict
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from data_cleaning import ColumnConfig, DataCleaning

class RandomForestModel_1:
    """
    Use dataset without categorical data.
    """
    
    def __init__(self):
        self.classification_params = {
            "n_estimators": 300,
            "max_depth": 8,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "bootstrap": True,
            "random_state": 42
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
        Training an Random Forest classification model
        default trained = True: There's already a trained model, so don't need to train again.
        """

        if trained == True:
            pass 
        else: 
            # train classification model
            if config is None:
                config = self.classification_params

            # set model config and train with fit function
            model = RandomForestClassifier(**config)
            model = model.fit(X_train, y_train)

            results = model.predict(X_test)
            report = classification_report(y_test, results)
            print("report:", report)

            # Save trained model
            with open('classification/models/randomforest-classification-1.pickle', 'wb') as f:
                pickle.dump(model, f)


if __name__ == "__main__":
    """
    Testing model 1
    """

    # ==================== Initiate Class ====================
    rfmodel = RandomForestModel_1() 
    titianic_training_config = ColumnConfig(
        int_cols = [
            "Survived", "Pclass", "Age", "SibSp", "Parch"
        ],
        float_cols = [
            "Fare"
        ],
        # category_cols = [
        #     "Sex", "Cabin", "Embarked"
        # ]
    )
    titianic_testing_config = ColumnConfig(
        int_cols = [
            "Pclass", "Age", "SibSp", "Parch"
        ],
        float_cols = [
            "Fare"
        ],
        # category_cols = [
        #     "Sex", "Cabin", "Embarked"
        # ]
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
    rfmodel._train_classification(
        X_train, 
        X_test,
        y_train, 
        y_test, 
        trained = True
    )

    # ==================== Test Model ====================
    try:
        with open('classification/models/randomforest-classification-1.pickle', 'rb') as f:
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


# class RandomForestModel_2:
#     """
#     Use dataset with categorical but encoded data.
#     ⚠️ If the unique values in a specific categorical columns are not same between training and testing dataset. It will cause Feature Mismatch issue
#     """
    
#     def __init__(self):
#         self.classification_params = {
#             "n_estimators": 300,
#             "max_depth": 8,
#             "min_samples_split": 5,
#             "min_samples_leaf": 2,
#             "max_features": "sqrt",
#             "bootstrap": True,
#             "random_state": 42
#         }

#     def _train_classification(
#             self, 
#             X_train: pd.DataFrame, 
#             X_test: pd.DataFrame,
#             y_train: pd.DataFrame,
#             y_test: pd.DataFrame,
#             config: Dict = None,
#             trained: bool = True
#     ):
#         """
#         Training an Random Forest classification model
#         default trained = True: There's already a trained model, so don't need to train again.
#         """

#         if trained == True:
#             pass 
#         else: 
#             # train classification model
#             if config is None:
#                 config = self.classification_params

#             # set model config and train with fit function
#             model = RandomForestClassifier(**config)
#             model = model.fit(X_train, y_train)

#             results = model.predict(X_test)
#             report = classification_report(y_test, results)
#             print("report:", report)

#             # Save trained model
#             with open('classification/models/randomforest-classification-2.pickle', 'wb') as f:
#                 pickle.dump(model, f)


# if __name__ == "__main__":
#     """
#     Testing model 2
#     """

#     # ==================== Initiate Class ====================
#     rfmodel = RandomForestModel_2() 
#     titianic_training_config = ColumnConfig(
#         int_cols = [
#             "Survived", "Pclass", "Age", "SibSp", "Parch"
#         ],
#         float_cols = [
#             "Fare"
#         ],
#         category_cols = [
#             "Sex", "Cabin", "Embarked"
#         ]
#     )
#     titianic_testing_config = ColumnConfig(
#         int_cols = [
#             "Pclass", "Age", "SibSp", "Parch"
#         ],
#         float_cols = [
#             "Fare"
#         ],
#         category_cols = [
#             "Sex", "Cabin", "Embarked"
#         ]
#     )

#     titianic_training_cleaner = DataCleaning(columns = titianic_training_config)
#     titianic_testing_cleaner = DataCleaning(columns = titianic_testing_config)

#     # ==================== Data Processing ====================
#     train_df = pd.read_csv("titanic/train_cleaned.csv")
#     cleaned_train_df = titianic_training_cleaner.clean_data(data=train_df)

#     # encode categorical columns, because categorical data is not accessible in scikit-learn python package.
#     cleaned_train_df = pd.get_dummies(
#         cleaned_train_df,
#         columns=titianic_training_config.category_cols,
#         dummy_na=True
#     )

#     answers = pd.read_csv("titanic/gender_submission.csv")


#     y = cleaned_train_df['Survived']
#     X = cleaned_train_df.drop(columns=['Survived'])

#     X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
#     X_test, X_valid, y_test, y_valid = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


#     # ==================== Training Model ====================
#     rfmodel._train_classification(
#         X_train, 
#         X_test,
#         y_train, 
#         y_test, 
#         trained = True
#     )

#     # ==================== Test Model ====================
#     try:
#         with open('classification/models/randomforest-classification-2.pickle', 'rb') as f:
#             model = pickle.load(f)

#     except FileNotFoundError:
#         print("Model not found. Please train the model first.")

#     test_df = pd.read_csv("titanic/test.csv")
#     cleaned_test_df = titianic_testing_cleaner.clean_data(data = test_df)
    
#     # encode categorical columns to match training data
#     cleaned_test_df = pd.get_dummies(
#         cleaned_test_df,
#         columns=titianic_training_config.category_cols,
#         dummy_na=True
#     )

#     predictions = model.predict(cleaned_test_df)

#     results = pd.DataFrame({
#         "PassengerId": test_df["PassengerId"],
#         "Predicted": predictions
#     })

#     comparison = results.merge(
#         answers,
#         on="PassengerId",
#         how="inner"
#     )

#     # np.where(condition, value_if_true, value_if_false)
#     comparison["check"] = np.where(
#         comparison["Predicted"] == comparison["Survived"],
#         "correct",
#         "wrong"
#     )

#     counts = comparison["check"].value_counts()
#     correct = counts["correct"]
#     wrong = counts["wrong"]
#     score = correct/(correct+wrong)

#     print("Correct Rate:", score)

