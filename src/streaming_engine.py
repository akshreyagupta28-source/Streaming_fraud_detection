import pandas as pd

from fraud_detector import FraudDetector


DATA_PATH = "dataset/processed/paysim_processed.csv"


def transaction_stream(file_path, chunksize=1000):

    for chunk in pd.read_csv(
        file_path,
        chunksize=chunksize
    ):

        for _, transaction in chunk.iterrows():

            yield transaction


if __name__ == "__main__":

    stream = transaction_stream(DATA_PATH)

    detector = FraudDetector()

    for i in range(10):

        transaction = next(stream)

        prediction, probability = detector.predict(
            transaction
        )

        print("Transaction:", i + 1)

        print(
            "Fraud Probability:",
            round(probability, 4)
        )

        if prediction == 1:
            print("Decision: FRAUD")
        else:
            print("Decision: LEGITIMATE")

        print("-" * 50)