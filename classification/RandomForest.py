
"""
1. Test this class with `python -m classification.RandomForest` in root

2. Random Forest wants features to be one of these:
    - int
    - float
    - bool
3. Because Random Forest cannot accept string or category columns, so we try dummy instead.
"""
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from typing import Dict
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from data_cleaning import ColumnConfig, DataCleaning

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class RandomForestModel:
    """
    Use dataset without categorical data.
    """
    
    def __init__(self):
        self.model_folder = "classification/models/"
        os.makedirs(self.model_folder, exist_ok=True)

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
            with open( self.model_folder + 'randomforest-classification.pickle', 'wb') as f:
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
    rfmodel = RandomForestModel() 
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
    rfmodel._train_classification(
        X_train, 
        X_test,
        y_train, 
        y_test, 
        trained = False
    )

    # ==================== Test Model ====================
    try:
        with open('classification/models/randomforest-classification.pickle', 'rb') as f:
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

    os.makedirs("classification/predict_results/", exist_ok=True)

    rfmodel._visualize_pca(
        X = cleaned_test_df, 
        predictions = predictions,
        save_path = "classification/predict_results/pca_visualization_randomForest.png",
        title = "PCA - Random Forest Predicted Classification (Test Set)"
    )
