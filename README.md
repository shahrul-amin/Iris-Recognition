# Project Title

Iris Species Classification using Machine Learning Models

## Description

This project involves using various machine learning models to classify the species of iris flowers based on their features. We utilize popular libraries such as pandas and numpy for data manipulation, matplotlib.pyplot and seaborn for data visualization, and scikit-learn for model training and evaluation. The models used include DecisionTreeClassifier, LogisticRegression, SVC, and KNeighborsClassifier. The dataset used is the famous Iris dataset.

## Getting Started

### Dependencies

* Python 3.x
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* warnings

### Installing

1. Clone the repository or download the project files.
2. Ensure you have the required libraries installed. You can install them using pip:
    ```
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```

### Executing program

1. Load the dataset:
    ```python
    import pandas as pd
    df = pd.read_csv("iris.csv")
    ```

2. Drop the 'Id' column:
    ```python
    df = df.drop(columns=["Id"])
    ```

3. Data visualization:
    ```python
    df["SepalLengthCm"].hist()
    df["SepalWidthCm"].hist()
    df["PetalLengthCm"].hist()
    df["PetalWidthCm"].hist()
    ```

4. Preprocess the data:
    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df["Species"] = le.fit_transform(df["Species"])
    ```

5. Split the data into training and testing sets:
    ```python
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=['Species'])
    y = df['Species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    ```

6. Train and evaluate models:
    ```python
    from sklearn.metrics import classification_report
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    import matplotlib.pyplot as plt

    def evaluationTest(model):
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))

        fig, ax = plt.subplots(figsize=(8, 5))
        cmp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=["class_1", "class_2", "class_3"])
        cmp.plot(ax=ax)
        plt.show()

    models = {
        'Decision Tree': DecisionTreeClassifier(),
        'Logistic Regression': LogisticRegression(),
        'Support Vector Machine Linear': SVC(kernel='linear', gamma=0.5, C=1.0),
        'Support Vector Machine rbf': SVC(kernel='rbf', gamma=0.5, C=1.0),
        'KNeighbors Classifier': KNeighborsClassifier()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        print(name)
        evaluationTest(model)
    ```
