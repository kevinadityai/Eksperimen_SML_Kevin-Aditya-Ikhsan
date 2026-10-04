"""Automated preprocessing for the Bank Marketing dataset.

Same steps as section 5 of Eksperimen_Kevin-Aditya-Ikhsan.ipynb, restructured into functions.

Usage (from the repository root):
    python preprocessing/automate_Kevin-Aditya-Ikhsan.py \
        --input bankmarketing_raw/bank-additional-full.csv \
        --output preprocessing/bankmarketing_preprocessing
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler

TARGET = "deposit"


def load_data(path):
    """Load the raw semicolon-separated CSV."""
    return pd.read_csv(path, sep=";")


def clean_data(df):
    """Encode the target, drop leakage/sentinel columns, duplicates and rare categories."""
    df = df.rename(columns={"y": TARGET})
    df[TARGET] = df[TARGET].map({"yes": 1, "no": 0})
    df = df.drop(columns=["duration", "pdays"])
    df = df.drop_duplicates()
    df = df[df["marital"] != "unknown"]
    df = df[df["education"] != "illiterate"]
    df = df[df["default"] != "yes"]
    return df


def split_data(df):
    """Stratified 80/20 train-test split."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def transform_features(X_train, X_test):
    """Scale numeric features (RobustScaler) and one-hot encode categorical ones, fitted on train only."""
    num_cols = X_train.select_dtypes(include=np.number).columns
    cat_cols = X_train.select_dtypes(include="object").columns

    scaler = RobustScaler()
    X_train_num = pd.DataFrame(scaler.fit_transform(X_train[num_cols]), columns=num_cols)
    X_test_num = pd.DataFrame(scaler.transform(X_test[num_cols]), columns=num_cols)

    encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    X_train_cat = pd.DataFrame(encoder.fit_transform(X_train[cat_cols]), columns=encoder.get_feature_names_out())
    X_test_cat = pd.DataFrame(encoder.transform(X_test[cat_cols]), columns=encoder.get_feature_names_out())

    return pd.concat([X_train_num, X_train_cat], axis=1), pd.concat([X_test_num, X_test_cat], axis=1)


def preprocess(raw_path, output_dir):
    """Run the full pipeline, save train.csv/test.csv to output_dir and return both DataFrames."""
    df = clean_data(load_data(raw_path))
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_t, X_test_t = transform_features(X_train, X_test)

    train_df = pd.concat([X_train_t, y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test_t, y_test.reset_index(drop=True)], axis=1)

    os.makedirs(output_dir, exist_ok=True)
    # lineterminator="\n" keeps the output byte-identical on Windows and Linux (GitHub Actions).
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False, lineterminator="\n")
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False, lineterminator="\n")
    return train_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the Bank Marketing dataset.")
    parser.add_argument("--input", default="bankmarketing_raw/bank-additional-full.csv")
    parser.add_argument("--output", default="preprocessing/bankmarketing_preprocessing")
    args = parser.parse_args()

    train_df, test_df = preprocess(args.input, args.output)
    print(f"Train: {train_df.shape}, Test: {test_df.shape}, saved to {args.output}")
