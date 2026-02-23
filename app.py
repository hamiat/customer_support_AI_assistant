import seaborn as sns
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

def createDataSet():
    titanic = sns.load_dataset("titanic")
    df = titanic[["survived", "pclass", "sex", "age", "fare", "embarked"]].dropna() 
    return df

def main():
    df = createDataSet()
    df = pd.get_dummies(df, drop_first=True)
    X = df.drop("survived", axis=1)
    y = df["survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 (primary metric):", f1_score(y_test, y_pred))
    print("AUC-ROC:", roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\n Classification Report:\n", classification_report(y_test, y_pred))

def kn():
    df = createDataSet()
    X = df.drop("survived", axis=1)
    y = df["survived"]

    X = pd.get_dummies(X, columns=["sex", "embarked"], drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsClassifier()

    param_grid = {
        "n_neighbors" : [1,3, 5, 7, 11, 13]
    }

    grid = GridSearchCV(
        knn,
        param_grid,
        cv=5,
        scoring="f1"
        )
    
    grid.fit(X_train_scaled, y_train)

    print("Best Param: ", grid.best_params_)
    print("Best F1 Score: ", grid.best_score_)


    


if __name__ == "__main__":
    #main()
    kn()