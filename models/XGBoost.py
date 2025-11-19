
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

class XGBoostModel:
    def __init__(self, objective, max_depth, learning_rate, subsample, colsample, num_estimators):
        self.model = self._build_model(objective, max_depth, learning_rate, subsample, colsample, num_estimators)

    def _build_model(self, objective, max_depth, learning_rate, subsample, colsample, num_estimators):
        """
        Example:
            model = xgb.XGBClassifier(
                # Objective
                objective = 'binary:logistic',      # or 'multi:softprob' for multi-class
                num_class = 3,                      # required only for multi-class tasks

                # Tree & Model Complexity
                max_depth = 6,
                n_estimators = 300,
                learning_rate = 0.001,

                # Subsampling
                subsample = 0.8,                    # row sampling
                colsample_bytree = 0.8,             # feature sampling

                # (Optional common parameters)
                min_child_weight = 1,
                gamma = 0,
                reg_alpha = 0,
                reg_beta = 1,
                random_state = 42
            )
        """
        model = xgb.XGBClassifier(
            objective=objective,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample,
            n_estimators=num_estimators,
            random_state=42
        )
        return model

    # ====== Training ======
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    # ====== Prediction ======
    def predict(self, X_test):
        return self.model.predict(X_test)

    # ====== Evaluation ======
    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds)
        return acc, report



# ===================== Model Testing Example ===================== 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
if __name__ == "__main__":

    # Load dataset
    df = pd.read_csv("titanic/train_cleaned.csv")

    y = df['Survived']
    X = df.drop(columns=['Survived'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = XGBoostModel(
        objective="binary:logistic",
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample=0.8,
        num_estimators=300
    )

    model.train(X_train, y_train)

    acc, report = model.evaluate(X_test, y_test)

    print("Accuracy:", acc)
    print("\nClassification Report:")
    print(report)