"""
1. Test this class with `python3 -m classification.RandomForest` in root

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

class randomForest:

    def __init__(self):

        self.MODEL_FOLDER      = "classification/models/"
        self.PCA_RESULT_FOLDER = "classification/predict_results/"

        self.RF_PARAMS = {
            "n_estimators": 300,
            "max_depth": 8,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "bootstrap": True,
            "random_state": 42
        }

        os.makedirs(self.MODEL_FOLDER, exist_ok=True)
        os.makedirs(self.PCA_RESULT_FOLDER, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _get_model(self):
        return RandomForestClassifier(**self.RF_PARAMS)

    def _model_path(self) -> str:
        return f"{self.MODEL_FOLDER}randomForest.pickle"

    # ------------------------------------------------------------------ #
    #  Main helpers                                                         #
    # ------------------------------------------------------------------ #

    def train(
        self,
        X_train : pd.DataFrame,
        X_test  : pd.DataFrame,
        y_train : pd.DataFrame,
        y_test  : pd.DataFrame,
        trained : bool = True
    ):
        """
        Training an SVM classification model
        default trained = True: There's already a trained model, so don't need to train again.
        """
        if trained:
            print(f"[TRAIN] Skipping — loading existing model from {self._model_path()}")
            return

        model = self._get_model()
        model.fit(X_train, y_train)

        report = classification_report(y_test, model.predict(X_test))
        print("TRAIN] Classification Report:\n", report)

        with open(self._model_path(), "wb") as f:
            pickle.dump(model, f)
        print(f"[TRAIN] Model saved to {self._model_path()}")

    def load(self):
        """ Load and return the persisted model. """
        try:
            with open(self._model_path(), "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"[LOAD] Model not found at {self._model_path()}. Please train the model first."
            )

    def evaluate(self, model, test_df: pd.DataFrame, cleaned_test_df: pd.DataFrame, answers: pd.DataFrame):
        """
        Run predictions and return correct rate.
        test_df        — raw dataframe, used to get PassengerId
        cleaned_test_df — cleaned features, used for prediction
        """
        # use reset_index to make sure two df have same index
        test_df         = test_df.reset_index(drop=True)
        cleaned_test_df = cleaned_test_df.reset_index(drop=True)
        predictions = model.predict(cleaned_test_df)

        results = pd.DataFrame({
            "PassengerId": test_df["PassengerId"],
            "Predicted"  : predictions
        })

        comparison = results.merge(answers, on="PassengerId", how="inner")

        comparison["check"] = np.where(
            comparison["Predicted"] == comparison["Survived"],
            "correct",
            "wrong"
        )

        counts  = comparison["check"].value_counts()
        correct = counts["correct"]
        wrong   = counts["wrong"]
        score   = correct / (correct + wrong)

        print("[EVALUATE] Correct Rate:", score)
        return predictions, score
    

    def visualize_pca(self, X: pd.DataFrame, predictions: np.ndarray):
        """
        Project features to 2D with PCA and plot predicted classes.
        """
        pca   = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        label_map = {1: "Survived",   0: "Not Survived"}
        colors    = {1: "steelblue",  0: "tomato"}

        plt.figure(figsize=(8, 6))
        for label in [0, 1]:
            mask = predictions == label
            plt.scatter(
                X_pca[mask, 0], X_pca[mask, 1],
                c=colors[label], label=label_map[label],
                alpha=0.6, edgecolors="white", linewidths=0.5
            )

        plt.title(f"PCA — Random Forest Predicted Classification")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
        plt.legend()
        plt.tight_layout()

        save_path = f"{self.PCA_RESULT_FOLDER}pca_randomForest.png"
        plt.savefig(save_path, dpi=150)
        print(f"[VISUALIZE] Plot saved to {save_path}")


if __name__ == "__main__":

    # ------------------------------------------------------------------ #
    #  Config                                                              #
    # ------------------------------------------------------------------ #
    TRAIN_CONFIG = ColumnConfig(
        int_cols   = ["Survived", "Pclass", "Age", "SibSp", "Parch"],
        float_cols = ["Fare"]
    )
    TEST_CONFIG = ColumnConfig(
        int_cols   = ["Pclass", "Age", "SibSp", "Parch"],
        float_cols = ["Fare"]
    )

    # ------------------------------------------------------------------ #
    #  Data                                                                #
    # ------------------------------------------------------------------ #
    train_df         = pd.read_csv("titanic/train_cleaned.csv")
    cleaned_train_df = DataCleaning(columns=TRAIN_CONFIG).clean_data(train_df)
    answers          = pd.read_csv("titanic/gender_submission.csv")

    y = cleaned_train_df["Survived"]
    X = cleaned_train_df.drop(columns=["Survived"])

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_test,  X_valid, y_test, y_valid = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    test_df         = pd.read_csv("titanic/test_cleaned.csv")
    cleaned_test_df = DataCleaning(columns=TEST_CONFIG).clean_data(test_df)

    # ------------------------------------------------------------------ #
    #  Train / Evaluate / Visualize                                         #
    # ------------------------------------------------------------------ #
    rfmodel = randomForest()

    try:
        rfmodel.train(X_train, X_test, y_train, y_test, trained=False)
        try: 
            model = rfmodel.load()
            try:
                predictions, score = rfmodel.evaluate(model, test_df, cleaned_test_df, answers)
                try:
                    rfmodel.visualize_pca(cleaned_test_df, predictions) 
                except Exception as e:
                    print(f"[VISUALIZE] Failed: {e}")
            except Exception as e:
                print(f"[EVALUATE] Failed: {e}")
        except FileNotFoundError as e:
            print(f"[LOAD] Failed: {e}")
    except Exception as e:
        print(f"[TRAIN] Failed: {e}")
        
