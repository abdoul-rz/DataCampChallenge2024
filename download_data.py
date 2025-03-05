import os
import hashlib
import pandas as pd
import kagglehub
import sklearn.model_selection as skms


def get_dataset():
    """Download the dataset from Kaggle."""
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    file_path = path + "/creditcard.csv"
    return file_path


def get_train_test(file_path):
    """Split the data into train and test sets."""
    dataset = pd.read_csv(file_path)

    # randomly sample 50% of the non-fraudulent transactions from
    dataset = dataset.drop(dataset[dataset.Class == 0].sample(frac=0.5, random_state=42).index)

    X, y = dataset.drop(columns=["Class"]), dataset["Class"]

    # a stratified split is used to ensure that the class distribution is
    # similar in both the training and testing sets
    X_train, X_test, y_train, y_test = skms.train_test_split(
        X, y, test_size=0.6, random_state=42, stratify=y
    )

    # combine the features and target into a single DataFrame
    X_train["Class"] = y_train
    X_test["Class"] = y_test

    # then save the new train data to disk
    X_train.to_csv('data/train.csv', index=False)
    X_test.to_csv('data/test.csv', index=False)


def download_data():
    """Download the data from Kaggle and split it into train and test sets."""
    file_path = get_dataset()
    get_train_test(file_path)
    # delete the downloaded file
    os.remove(file_path)


if __name__ == "__main__":
    download_data()