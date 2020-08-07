import pandas as pd
import numpy as np
import warnings
import sklearn.metrics as mt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# To avoid warnings
warnings.filterwarnings('ignore')


def read_data(path):
    """Read and return data."""
    data = pd.read_csv(path)
    return data


def data_prepare(dataset):
    """Puts data in order in a few steps.

    1. Delete unused columns
    2. Replace NaN's with means and most frequent
    3. Replace str values with ints
    4. Depersonalization of some data, bringing them to a vector form

    Returns prepared dataset.

    """
    # Delete unused columns
    unused_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Fare']
    data = dataset.drop(unused_columns, axis=1)

    # Replace NaN's with means...
    feature_list_1 = ['Age']
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    data[feature_list_1] = imputer.fit_transform(data[feature_list_1].astype('float64'))

    # ...and most frequent
    feature_list_2 = ['Survived', 'Pclass', 'SibSp', 'Parch']
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    data[feature_list_2] = imputer.fit_transform(data[feature_list_2].astype('float64'))

    # Replace str values with ints
    label_encoder_sex = LabelEncoder()
    data['Sex'] = label_encoder_sex.fit_transform(data['Sex'].astype(str))
    label_encoder_embarked = LabelEncoder()
    data['Embarked'] = label_encoder_embarked.fit_transform(data['Embarked'].astype(str))

    # Depersonalization of some data, bringing them to a vector form
    # e.g. for Sex column will be created Sex_0 and Sex_1 columns
    categorical_feature_list = ['Sex', 'Embarked', 'Pclass']
    for feature in categorical_feature_list:
        data[feature] = pd.Categorical(data[feature])
        data_dummies = pd.get_dummies(data[feature], prefix=feature)
        data = pd.concat([data, data_dummies], axis=1)
        data = data.drop(labels=[feature], axis=1)

    return data


def get_x_and_y(data):
    """Splits dataset into feature matrix X and vector valid answers y."""
    feature_list = ['Age', 'SibSp', 'Parch',
                    'Sex_0', 'Sex_1', 'Embarked_0',
                    'Embarked_1', 'Embarked_2',
                    'Embarked_3', 'Pclass_1.0',
                    'Pclass_2.0', 'Pclass_3.0']
    X = data[feature_list]
    y = data[['Survived']]
    return X, y


def cross_validation(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    return X_train, X_test, y_train, y_test


def train_KNN(X, y):
    """Train KNN."""
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X, y)
    return knn


def train_LogReg(X, y):
    """Train LogReg."""
    lg = LogisticRegression()
    lg.fit(X, y)
    return lg


def predict_KNN(knn_model, passenger):
    """Returns KNN pediction."""
    prediction = knn_model.predict(passenger)

    if prediction:
        print('KNN: Passenger save!')
    else:
        print('KNN: Passenger die :(')


def predict_LogReg(lg_model, passenger):
    """Returns LogReg pediction."""
    prediction = lg_model.predict(passenger)

    if prediction == 1:
        print('LogReg: Passenger save!')
    else:
        print('LogReg: Passenger die :(')


def model_passenger(X):
    """Modeling passenger."""
    params = []

    for column in X.columns:
        param = int(input(f'{column}: '))
        params.append(param)
    passenger = np.array(params).reshape(1, -1)

    return passenger


def print_metrics(knn_model, lg_model, X_test, y_test):
    """Print metric results of KNN and LogReg."""
    prediction = knn_model.predict(X_test)
    accuracy = mt.accuracy_score(y_test, prediction)
    print(f'KNN metric: {100 * accuracy}')

    prediction = lg_model.predict(X_test)
    accuracy = mt.accuracy_score(y_test, prediction)
    print(f'LogReg metric: {100 * accuracy}')


def test_KNN_and_LogReg(X, y):
    """KNN and LogReg accuracy test."""
    X_train, X_test, y_train, y_test = cross_validation(X, y)

    # Train models
    knn_model = train_KNN(X_train, y_train)
    lg_model = train_LogReg(X_train, y_train)

    print_metrics(knn_model, lg_model, X_test, y_test)


def play_with_own_passenger(X, y):
    """Model user's passenger and predicts."""
    # Train models
    knn_model = train_KNN(X, y)
    lg_model = train_LogReg(X, y)

    # Model passenger
    passenger = model_passenger(X)

    # Predictions
    predict_KNN(knn_model, passenger)
    predict_LogReg(lg_model, passenger)


def main():
    # Read data
    dataset = read_data('data/train.csv')

    # Data prepare
    dataset = data_prepare(dataset)

    # Get feature matrix and vector valid answers
    X, y = get_x_and_y(dataset)

    test_KNN_and_LogReg(X, y)
    play_with_own_passenger(X, y)


if __name__ == '__main__':
    main()
