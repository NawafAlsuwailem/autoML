from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

from preprocessing import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

def get_ann(X_train, y_train):
    numberOfColumns = X_train.shape[1]
    classifier = Sequential()
    classifier.add(Dense(output_dim=numberOfColumns * 8, init='uniform', activation='relu', input_dim=numberOfColumns))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(output_dim=numberOfColumns * 6, init='uniform', activation='relu'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    classifier.fit(X_train, y_train, batch_size=10, nb_epoch=1)
    return classifier

def get_rf(X_train, y_train):
    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    # Train the model on training data
    rf.fit(X_train, y_train)
    return rf


def get_knn(X_train, y_train):
    knn_params = {
        'n_neighbors': [5, 40],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']}
    knn_gridsearch = GridSearchCV(KNeighborsClassifier(),
                                  knn_params,
                                  n_jobs=-1, cv=5)  # try verbose!
    knn_gridsearch.fit(X_train, y_train)
    best_knn = knn_gridsearch.best_estimator_
    return best_knn


def get_baseline(dataframe, target_feature):
    baseline = dataframe[target_feature].value_counts(normalize=True)
    baseline = baseline.to_frame().T
    return baseline


def get_ypredict(classifier, X_test):
    confRate = 0.5
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > confRate)
    return y_pred


def get_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    return cm


def get_acc_score(y_pred, y_test):
    return accuracy_score(y_test, y_pred)


def get_classification_report(y_test, ann_ypredict):
    report = classification_report(y_test, ann_ypredict, output_dict = True)
    report = pd.DataFrame(report).T
    return report


def cm2df(cm, labels):
    df = pd.DataFrame()
    # rows
    for i, row_label in enumerate(labels):
        rowdata={}
        # columns
        for j, col_label in enumerate(labels):
            rowdata[col_label]=cm[i,j]
        df = df.append(pd.DataFrame.from_dict({row_label:rowdata}, orient='index'))
    return df[labels]


# dataframe = pd.read_csv("../UPLOAD_FOLDER/iris.csv")
# dataframe.columns = [i.lower() for i in dataframe.columns]
# for column in dataframe.columns:
#     if len(dataframe[column].unique()) == dataframe.shape[0]:
#         del dataframe[column]
# columns = dataframe.columns
# target_feature_values = dataframe["species"].unique()
# print(target_feature_values)
# baseline = get_baseline(dataframe, "species")
#
# categorical_columns = get_categorical_data(dataframe).columns
# dataframe = deal_null(dataframe, "remove")
# dataframe = apply_label_encoder(dataframe)
# dataframe = deal_outlier(dataframe, "remove")
# X = define_independent_variables(dataframe, "species")
# y = define_dependant_variable(dataframe, "species")
# X = get_selected_features(X, ["sepallengthcm", "sepalwidthcm"])
# X = apply_one_hot_encoder(X, categorical_columns)
# X_train, X_test, y_train, y_test = split_data_set(X, y, 2.5)
# X_train, X_test = apply_standard_scaling(X_train, X_test)
#
#
# # classifier
# ann = get_ann(X_train, y_train)
# knn = get_knn(X_train, y_train)
#
# # classifier info
# ann_ypredict = get_ypredict(ann, X_test)
# knn_ypredict = get_ypredict(knn, X_test)
#
#
# ann_cm = get_confusion_matrix(y_test, ann_ypredict)
# knn_cm = get_confusion_matrix(y_test, knn_ypredict)
#
#
# ann_cm = cm2df(ann_cm, target_feature_values)
#
#
#
# ann_acc = get_acc_score(ann_ypredict, y_test)
# knn_acc = get_acc_score(knn_ypredict, y_test)
#
#
#
# ann_report = get_classification_report(y_test, ann_ypredict)
# knn_report = get_classification_report(y_test, knn_ypredict)
#
