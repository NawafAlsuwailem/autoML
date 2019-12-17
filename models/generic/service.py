def get_categorical_data(dataframe):
    return dataframe.select_dtypes(include='object')