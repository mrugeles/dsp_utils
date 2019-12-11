import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from category_encoders.ordinal import OrdinalEncoder 

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class Encoder():
    def one_hot_encode(self, df, column):
        df = df.join(pd.get_dummies(df[column], prefix=column))
        return df.drop([column], axis = 1)  

    def label_encode(self, df, column):
        le = preprocessing.LabelEncoder()
        values = list(df[column].values)
        le.fit(values)
        df[column] = le.transform(values)
        return df

    def ordinal_encode(self, df, column, map_list):
        mapping_dict = [{'col': column, 'mapping': map_list}]
        enc = OrdinalEncoder(mapping = mapping_dict)

        return enc.fit_transform(df[column])

    def scale_normalize(self, df, columns):
        df[columns] = MinMaxScaler().fit_transform(df[columns])
        for column in columns:
            df[column] = df[column].apply(lambda x: np.log(x + 1))
        return df

    def categorize(self, df, column, bins):
        data = pd.cut(np.array(df[column]),  bins=bins)
        data = pd.Series(data)
        data = pd.DataFrame(data, columns=[f'{column}_Range'])
        data = data[f'{column}_Range'].apply(lambda value: str(value).replace('(', '').replace(']', '').replace(', ', '_'))

        df = df.join(pd.DataFrame(data, columns=[f'{column}_Range']))
        df = df.join(pd.get_dummies(df[f'{column}_Range']))
        df = df.drop([column], axis = 1)
        return df.drop([f'{column}_Range'], axis = 1)

    def scale_range(self, X):
        return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    def scale_max_clipping(self, X, clip):
        return X.apply(lambda x: x if x < clip else clip)

    def scale_min_clipping(self, X, clip):
        return X.apply(lambda x: x if x > clip else clip)

    def scale_log(self, X):
        return X.apply(lambda x: np.log(x + 1))

    def scale_z_score(self, X):
        return X.apply(lambda x: (x - X.mean() / X.std()))

    def plot_normalizers(self, X, clip = 10):
        fig, ax = plt.subplots(1, 5, figsize=(17, 2))
        X_range = self.scale_range(X)
        X_clipped = self.scale_max_clipping(X, clip)
        X_log_scaled = self.scale_log(X)
        X_z_score = self.scale_z_score(X)

        ax[0].hist(X)
        ax[1].hist(X_range)
        ax[1].set_title('Scale Range')
        ax[2].hist(X_clipped)
        ax[2].set_title('Clipped')
        ax[3].hist(X_log_scaled)
        ax[3].set_title('Log')
        ax[4].hist(X_z_score)
        ax[4].set_title('Z-Score')