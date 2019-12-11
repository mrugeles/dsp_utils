
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import pandas as pd
import numpy as np

class PlotUtil():

    def plot_missing_values(self, dataset):
        zero_df = dataset.isnull().mean().to_frame(name = 'missing')
        non_zero_df = dataset.notnull().mean().to_frame(name = 'not_missing')
        missing_values = pd.concat([zero_df, non_zero_df], axis=1, sort=False)
        missing_values['total'] = missing_values['missing'] + missing_values['not_missing'] 
        print(zero_df[zero_df['missing'] > 0])
        
        plt.figure(figsize=(15,5))

        features = list(missing_values.index)
        width = .7
        ind = np.arange(len(features)) 

        plot_missing = plt.bar(ind, missing_values['missing'], width, color = 'red', alpha = 0.3)
        plot_not_missing = plt.bar(ind, missing_values['not_missing'], width, color = 'green', alpha = 0.3, bottom=missing_values['missing'])

        plt.ylabel('%')
        plt.title('% Missing values')
        plt.xticks(ind, tuple(features))
        plt.yticks(np.arange(0, .8, 1))
        plt.legend((plot_missing[0], plot_not_missing[0]), ('Missing', 'Not missing'), bbox_to_anchor=(1.05, 1), borderaxespad=0., loc='upper right')
        plt.xticks(rotation=45, ha='right')

        plt.show()