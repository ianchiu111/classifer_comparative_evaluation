
"""
1. Test this class with `python -m classification.SVM` in root
"""
import numpy as np
import pandas as pd
import pickle
from typing import Dict
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from data_cleaning import ColumnConfig, DataCleaning

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class SVM_1:
    """
    Linear function
    """
    
    def __init__(self):
        self.classification_params = {
            "C" : 1, 
            "max_iter" : 10000
        }

    def _train_classification(
            self, 
            X_train: pd.DataFrame, 
            X_test: pd.DataFrame,
            y_train: pd.DataFrame,
            y_test: pd.DataFrame,
            config: Dict = None,
            trained: bool = False
    ):
        """
        Training an SVM classification model
        default trained = True: There's already a trained model, so don't need to train again.
        """

        if trained == True:
            pass 
        else: 
            # train classification model
            if config is None:
                config = self.classification_params
            
            # set model config and train with fit function
            model = svm.LinearSVC(**config)
            model = model.fit(X_train, y_train)
            
            # 使用訓練資料預測分類
            results = model.predict(X_test)
            report = classification_report(y_test, results)
            print("report:", report)

            with open('classification/models/svm-classification-1.pickle', 'wb') as f:
                pickle.dump(model, f)

    def _visualize_pca(
            self,
            X: pd.DataFrame,
            predictions: np.ndarray,
            save_path: str,
            title: str
    ):
        """
        Visualize classification results using PCA (2D projection).
        """
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        label_map = {1: "Survived", 0: "Not Survived"}
        colors    = {1: "steelblue", 0: "tomato"}

        plt.figure(figsize=(8, 6))

        for label in [0, 1]:
            mask = predictions == label
            plt.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                c=colors[label],
                label=label_map[label],
                alpha=0.6,
                edgecolors='white',
                linewidths=0.5
            )

        plt.title(title)
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"PCA plot saved to: {save_path}")

if __name__ == "__main__":

    # ==================== Initiate Class ====================
    svmmodel = SVM_1()
    titianic_training_config = ColumnConfig(
        int_cols = [
            "Survived", "Pclass", "Age", "SibSp", "Parch"
        ],
        float_cols = [
            "Fare"
        ],
        # category_cols = [
        #     "Sex", "Embarked"
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
        #     "Sex", "Embarked"
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
    svmmodel._train_classification(
        X_train, 
        X_test,
        y_train, 
        y_test, 
        trained = False
    )

    # ==================== Test Model ====================
    try: 
        with open('classification/models/svm-classification-1.pickle', 'rb') as f:
            model = pickle.load(f)

    except FileNotFoundError:
        print("Model not found. Please train the model first.")

    test_df = pd.read_csv("titanic/test_cleaned.csv")
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

    # ==================== PCA Visualization ====================
    pca = PCA(n_components=2)

    # Use the test features that were predicted
    X_pca = pca.fit_transform(cleaned_test_df)

    # Map numeric predictions to labels
    label_map = {1: "Survived", 0: "Not Survived"}
    colors = {1: "steelblue", 0: "tomato"}

    plt.figure(figsize=(8, 6))

    for label in [0, 1]:
        mask = predictions == label
        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=colors[label],
            label=label_map[label],
            alpha=0.6,
            edgecolors='white',
            linewidths=0.5
        )

    plt.title("PCA - XGBoost Predicted Classification (Test Set)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("titanic/pca_visualization_svm_1.png", dpi=150)

    svmmodel._visualize_pca(
        X = cleaned_test_df, 
        predictions = predictions,
        save_path = "titanic/pca_visualization_svm_1.png",
        title = "PCA - SVM Predicted Classification (Test Set)"
    )

# class SVM_2:
#     """
#     Non-Linear function
#     """
    
#     def __init__(self):
#         self.classification_params = {
#             "kernel": 'poly', 
#             "degree": 3, 
#             "gamma": 'auto',
#             "C": 1
#         }

#     def _train_classification(
#             self, 
#             X_train: pd.DataFrame, 
#             X_test: pd.DataFrame,
#             y_train: pd.DataFrame,
#             y_test: pd.DataFrame,
#             config: Dict = None,
#             trained: bool = False
#     ):
#         """
#         Training an SVM classification model
#         default trained = True: There's already a trained model, so don't need to train again.
#         """

#         if trained == True:
#             pass 
#         else: 
#             # train classification model
#             if config is None:
#                 config = self.classification_params
            
#             # set model config and train with fit function
#             model = svm.SVC(**config)
#             model = model.fit(X_train, y_train)
            
#             # 使用訓練資料預測分類
#             results = model.predict(X_test)
#             report = classification_report(y_test, results)
#             print("report:", report)

#             with open('classification/models/svm-classification-2.pickle', 'wb') as f:
#                 pickle.dump(model, f)

#     def _visualize_pca(
#             self,
#             X: pd.DataFrame,
#             predictions: np.ndarray,
#             save_path: str,
#             title: str
#     ):
#         """
#         Visualize classification results using PCA (2D projection).
#         """
#         pca = PCA(n_components=2)
#         X_pca = pca.fit_transform(X)

#         label_map = {1: "Survived", 0: "Not Survived"}
#         colors    = {1: "steelblue", 0: "tomato"}

#         plt.figure(figsize=(8, 6))

#         for label in [0, 1]:
#             mask = predictions == label
#             plt.scatter(
#                 X_pca[mask, 0],
#                 X_pca[mask, 1],
#                 c=colors[label],
#                 label=label_map[label],
#                 alpha=0.6,
#                 edgecolors='white',
#                 linewidths=0.5
#             )

#         plt.title(title)
#         plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
#         plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(save_path, dpi=150)
#         print(f"PCA plot saved to: {save_path}")

# if __name__ == "__main__":

#     # ==================== Initiate Class ====================
#     svmmodel = SVM_2()
#     titianic_training_config = ColumnConfig(
#         int_cols = [
#             "Survived", "Pclass", "Age", "SibSp", "Parch"
#         ],
#         float_cols = [
#             "Fare"
#         ],
#         # category_cols = [
#         #     "Sex", "Embarked"
#         # ]
#     )
#     titianic_testing_config = ColumnConfig(
#         int_cols = [
#             "Pclass", "Age", "SibSp", "Parch"
#         ],
#         float_cols = [
#             "Fare"
#         ],
#         # category_cols = [
#         #     "Sex", "Embarked"
#         # ]
#     )

#     titianic_training_cleaner = DataCleaning(columns = titianic_training_config)
#     titianic_testing_cleaner = DataCleaning(columns = titianic_testing_config)

#     # ==================== Data Processing ====================
#     train_df = pd.read_csv("titanic/train_cleaned.csv")
#     cleaned_train_df = titianic_training_cleaner.clean_data(data = train_df)
#     answers = pd.read_csv("titanic/gender_submission.csv")

#     y = cleaned_train_df['Survived']
#     X = cleaned_train_df.drop(columns=['Survived'])

#     X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
#     X_test, X_valid, y_test, y_valid = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


#     # ==================== Training Model ====================
#     svmmodel._train_classification(
#         X_train, 
#         X_test,
#         y_train, 
#         y_test, 
#         trained = True
#     )

#     # ==================== Test Model ====================
#     try: 
#         with open('classification/models/svm-classification-2.pickle', 'rb') as f:
#             model = pickle.load(f)

#     except FileNotFoundError:
#         print("Model not found. Please train the model first.")

#     test_df = pd.read_csv("titanic/test_cleaned.csv")
#     cleaned_test_df = titianic_testing_cleaner.clean_data(data = test_df)

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

#     svmmodel._visualize_pca(
#         X = cleaned_test_df, 
#         predictions = predictions,
#         save_path = "titanic/pca_visualization_svm_2.png",
#         title = "PCA - SVM Predicted Classification (Test Set)"
#     )


