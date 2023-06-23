import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

if __name__ == '__main__':
    heart_data = pd.read_csv("heart.csv")

    heart_data_features = heart_data.iloc[:, :-1]
    hdt_cols = heart_data_features.columns
    scaler = StandardScaler()
    heart_data_features = scaler.fit_transform(heart_data_features)
    heart_data_features = pd.DataFrame(heart_data_features, columns=hdt_cols)

    heart_data_target = heart_data.loc[:, "output"]
    X_train, X_test, y_train, y_test = train_test_split(heart_data_features, heart_data_target, test_size=0.25, random_state=32)

    accuracy_scores = []
    best_model = KNeighborsClassifier
    best_accuracy = 0
    best_k = 0
    best_predictions = []
    for k in range(1, 32):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, predictions)
        accuracy_scores.append(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
            best_model = knn
            best_predictions = predictions

    print("Best accuracy: ", best_accuracy, " with k = ", best_k, " neighbors")

    # menu
    print("1. Enter data for prediction")
    print("2. Exit")
    choice = int(input("Enter your choice: "))
    while choice != 2:
        age = int(input("Enter age: "))
        sex = int(input("Enter sex (1 male, 0 female): "))
        cp = int(input("Enter chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal pain, "
                       "3: asymptomatic): "))
        trtstbps = int(input("Enter resting blood pressure: "))
        chol = int(input("Enter serum cholesterol in mg/dl: "))
        fbs = int(input("Enter fasting blood sugar > 120 mg/dl (1 true, 0 false): "))
        restecg = int(input("Enter resting electrocardiograph results (0: normal, 1: having ST-T wave abnormality, "
                            "2: showing probable or definite left ventricular hypertrophy by Estes' criteria): "))
        thalachh = int(input("Enter maximum heart rate achieved: "))
        exng = int(input("Enter exercise induced angina (1 yes, 0 no): "))
        oldpeak = float(input("Enter ST depression induced by exercise relative to rest: "))
        slope = int(input("Enter the slope of the peak exercise ST segment (0: up-sloping, 1: flat, 2: down-sloping): "))
        caa = int(input("Enter number of major vessels (0-3) colored by fluoroscopy: "))
        thal = int(input("Enter thalassemia (0: normal, 1: fixed defect, 2: reversible defect): "))

        features = [age, sex, cp, trtstbps, chol, fbs, restecg, thalachh, exng, oldpeak, slope, caa, thal]

        features = pd.DataFrame([features], columns=cols)

        res = best_model.predict(features)

        if res == 0:
            print("Less likely to experience a heart attack.")
        else:
            print("More likely to experience a heart attack.")

        print("1. Enter data for prediction")
        print("2. Exit")
        choice = int(input("Enter your choice: "))