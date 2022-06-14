import pandas as pd
from sklearn .preprocessing import LabelEncoder
import seaborn as sns


class DataProcessor:

    """
    Data Porcessor that reformats data and gives statistical insigts 
    """

    def __init__(self, dataset, categorical_flag = False):
        self.dataset = dataset
        self.categorical_flag = categorical_flag


   
    def load_data(self):
        """
        load the dataset into a dataframe

        :param dataset: <str> the name of the dataset
        :return: <DataFrame> returns the dataset as dataframe
        """
        try:
            df = pd.read_csv(self.dataset,sep=' ',header =None )
        except: 
            print('The dataset needs to be in csv format')

        if self.categorical_flag: 
            df = self.categorical_encoder(df)

        return df


    def categorical_encoder(self, df):
        """
        encode all categorical variables of a given dataframe

        :param df: <DataFrame> the given dataframe
        :return: <Dataframe> returns the dataframe with all the columns as numerical variables
        """

        label_encoder = LabelEncoder()

        for i in range(len(df.columns)):
            if df[i].dtype != 'int64': df[i] = label_encoder.fit_transform(df[i])

        return df


    def count_null(self, df):
        """
        count all the null values of the df

        :param df: <DataFrame> the given dataframe
        :return: <list> returns all the columns and the correpsonding number of null values
        """

        info = df.isna().sum()
        return info

    
    def stats_info(self, df):
        """
        give the statistical information for all the numeric variables

        :param df: <DataFrame> the given dataframe
        :return: <list> returns the list with the statistical information
        """
        numerical_columns = []

        for i in range(len(df.columns)):
            if df[i].dtype == 'int64': numerical_columns.append(df.columns[i])

        stats = df[numerical_columns].describe()
        return stats

    
    def plot_dist(self, df, column):
        """
        plot the distribution

        :param df: <DataFrame> the given dataframe
        :return: <list> returns the list with the statistical information
        """
        if df[column].dtype == 'int64':
            sns.boxplot(df[column])
        elif df[column].dtype != 'int64':
            sns.countplot(df[column])

        return


if __name__ == '__main__':

    processor = DataProcessor("data/german/german.data")

    df = processor.load_data()
    stats = processor.stats_info(df)
    processor.plot_dist(df,1)
    
    print(stats)

