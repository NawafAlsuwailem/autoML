from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np
from scipy import stats
from generic.service import get_categorical_data

sc = StandardScaler()
index_for_cat_data = []
names_for_cat_data = []


# deal with nulls - 2
def deal_null(dataframe, pref):
    if pref == "keep":
        return dataframe.apply(lambda x: x.fillna(x.mean()) if x.dtypes == int or x.dtypes == float else x.fillna(x.mode()[0]))
    else:
        return dataframe.dropna()


def deal_outlier(dataframe, pref):
    if pref == "remove":
        in_col_list = []
        out_col_list = []

        columns = dataframe.keys()
        for col in columns:
            column = dataframe[col]
            if dataframe.groupby(column).count().shape[0] > 2:
                in_col_list.append(column)
            else:
                out_col_list.append(column)

        in_df = pd.DataFrame(in_col_list).T
        out_df = pd.DataFrame(out_col_list).T


        threshold = 3
        z = np.abs(stats.zscore(in_df))
        in_df = in_df[(z < threshold).all(axis=1)]

        filt_df = pd.concat([in_df, out_df], axis=1)
        filt_df = filt_df.dropna()
        return filt_df
    else:
        return dataframe


def get_selected_features(X, selected_features):
    tempX = []
    for col in selected_features:
        column = X[col]
        tempX.append(column)
    X = pd.DataFrame(tempX).T
    return X


# independent variables - 3
def define_independent_variables(dataframe, target):
    dictionary = {}
    for col in dataframe.keys():
        if col != target:
            dictionary[col] = dataframe[col]
    X =pd.DataFrame.from_dict(dictionary)
    return X


# dependant variable - 4
def define_dependant_variable(dataframe, target):
    y = dataframe[target]
    y = pd.DataFrame(y, columns=[target])
    return y


# Encoding categorical data - 5
def apply_label_encoder(X):
    le = LabelEncoder()
    categorical_columns = get_categorical_data(X).columns
    try:
        for col in categorical_columns:
            # index = X.columns.get_loc(col)
            # index_for_cat_data.append(index)
            names_for_cat_data.append(col)
        X[categorical_columns] = X[categorical_columns].apply(lambda col: le.fit_transform(col))
    except:
        pass
    return X


# def get_cate_index():


# Splitting the dataset into the Training set and Test set - 6
def apply_one_hot_encoder(X, categorical_columns):
    for col in categorical_columns:
        if col in X:
            index = X.columns.get_loc(col)
            index_for_cat_data.append(index)
    try:
        for i in index_for_cat_data:
            print(i)
        onehotencoder = OneHotEncoder(categorical_features=[index_for_cat_data])
        X = onehotencoder.fit_transform(X).toarray()
        X = X[:, :]

    except:
        pass
    return X


# split dataset into training and testing - 7
def split_data_set(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_size)/10, random_state=0)
    return X_train, X_test, y_train, y_test


# apply dtandard scaling - 8
def apply_standard_scaling(X_train, X_test):
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test





