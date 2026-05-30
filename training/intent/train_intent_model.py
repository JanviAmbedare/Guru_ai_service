import os
import re
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from sklearn.linear_model import (
    LogisticRegression
)

from sklearn.model_selection import (
    train_test_split
)

from sklearn.metrics import (
    classification_report,
    accuracy_score
)


# =====================================
# CONFIG
# =====================================

DATASET_PATH = (
    "training/intent/augmented_dataset.csv"
)

MODEL_OUTPUT_PATH = (
    "models/exported/"
    "intent_pipeline.pkl"
)


# =====================================
# TEXT CLEANING
# =====================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        "",
        text
    )

    text = text.strip()

    return text


# =====================================
# SIMPLE DATA AUGMENTATION
# =====================================

def augment_text(text):

    augmented = []

    augmented.append(text)

    augmented.append(
        text.lower()
    )

    augmented.append(
        text.upper()
    )

    augmented.append(
        text.capitalize()
    )

    return list(set(augmented))


# =====================================
# LOAD DATASET
# =====================================

def load_dataset():

    if not os.path.exists(
        DATASET_PATH
    ):

        raise FileNotFoundError(
            f"Dataset not found: "
            f"{DATASET_PATH}"
        )

    df = pd.read_csv(
        DATASET_PATH
    )

    if "text" not in df.columns:

        raise Exception(
            "Dataset must contain "
            "'text' column"
        )

    if "intent" not in df.columns:

        raise Exception(
            "Dataset must contain "
            "'intent' column"
        )

    texts = []
    labels = []

    for _, row in df.iterrows():

        cleaned = clean_text(
            str(row["text"])
        )

        augmented_texts = (
            augment_text(cleaned)
        )

        for text in augmented_texts:

            texts.append(text)

            labels.append(
                row["intent"]
            )

    return texts, labels


# =====================================
# TRAIN MODEL
# =====================================

def train_model():

    print(
        "🚀 Loading dataset..."
    )

    texts, labels = (
        load_dataset()
    )

    print(
        f"✅ Samples loaded: "
        f"{len(texts)}"
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            texts,
            labels,
            test_size=0.2,
            random_state=42
        )
    )

    print(
        "🚀 Training model..."
    )

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=5000
                )
            ),

            (
                "classifier",
                LogisticRegression(
                    max_iter=1000
                )
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    print(
        "✅ Training completed"
    )

    # ==========================
    # EVALUATION
    # ==========================

    predictions = pipeline.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"✅ Accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    # ==========================
    # SAVE MODEL
    # ==========================

    os.makedirs(
        "models/exported",
        exist_ok=True
    )

    joblib.dump(
        pipeline,
        MODEL_OUTPUT_PATH
    )

    print(
        f"✅ Model saved: "
        f"{MODEL_OUTPUT_PATH}"
    )


# =====================================
# MAIN
# =====================================

if __name__ == "__main__":

    train_model()