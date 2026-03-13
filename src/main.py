# ===============================
# Heart Disease Prediction System
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


# ===============================
# 🔹 Normalization + Split
# ===============================
def normalization(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


# ===============================
# 🔹 Visualization
# ===============================
def visualization(df):

    total_counts = df["condition"].value_counts()
    heart_sick = total_counts[1]
    not_sick = total_counts[0]

    print("Heart Disease Dataset Analysis:")
    print("Patients with heart disease:", heart_sick)
    print("Healthy patients:", not_sick)
    print("Total patients:", heart_sick + not_sick)

    plt.figure(figsize=(10, 6))

    categories = ["Not Sick", "Sick"]
    counts = [not_sick, heart_sick]
    colors = ["lightblue", "lightcoral"]

    plt.bar(categories, counts, color=colors, edgecolor="black")
    plt.title("Heart Disease Distribution")

    for i, count in enumerate(counts):
        plt.text(i, count + 5, str(count), ha="center")

    plt.show()


# ===============================
# 🔹 Model (Stacking)
# ===============================
def build_model(X_train, y_train, X_test, y_test):

    base_models = [
        ("lr", LogisticRegression()),
        ("rf", RandomForestClassifier(n_estimators=100)),
        ("svm", SVC(probability=True)),
        ("et", ExtraTreesClassifier(n_estimators=100)),
    ]

    meta_model = ExtraTreesClassifier(n_estimators=100)

    model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return model, y_pred


# ===============================
# 🔹 Confusion Matrix
# ===============================
def plot_confusion(y_test, y_pred):

    sns.heatmap(confusion_matrix(y_test, y_pred),
                annot=True, fmt="d", cmap="Greens")

    plt.title("Stack Model - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# ===============================
# 🔹 MAIN
# ===============================

if __name__ == "__main__":

    # Load dataset
    df = pd.read_csv("data/heart_cleveland_upload.csv")

    X = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak',
            'slope', 'ca', 'thal']]

    y = df["condition"]

    # Steps
    X_train, y_train, X_test, y_test = normalization(X, y)

    visualization(df)

    model, y_pred = build_model(X_train, y_train, X_test, y_test)

    plot_confusion(y_test, y_pred)
