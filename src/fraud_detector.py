import joblib
import pandas as pd


MODEL_PATH = "models/xgboost.pkl"


class FraudDetector:

    def __init__(self, model_path=MODEL_PATH):
        self.model = joblib.load(model_path)

    def predict(self, transaction):

        data = pd.DataFrame([transaction])

        feature_columns = [
            "step",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "balance_change_orig",
            "balance_change_dest",
            "amount_to_orig_balance",
            "type_CASH_IN",
            "type_CASH_OUT",
            "type_DEBIT",
            "type_PAYMENT",
            "type_TRANSFER"
        ]

        features = data[feature_columns]

        probability = self.model.predict_proba(features)[0][1]

        prediction = int(probability >= 0.5)

        return prediction, probability