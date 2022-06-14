import tensorflow_decision_forests as tfdf
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from preprocessing import DataProcessor

class TfdfClassifier:
    """
    Trains TensorFlow Decision Forests for a dataframe with a given target
            ***Runs only on linux***
    """

    def __init__(self, df, target):
        self.df = df
        self.tager = target

        return


    def train_classifier(self):
        """
        train a tfdf model 

        :return: <h5> returns the model
        :return: <tensor slice> returns the test dataset
        """
        try:
            X_train, X_test = train_test_split(self.df, test_size=0.2, random_state=42, 
                                            stratify=self.df[self.tager],shuffle=True)


            train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(X_train, label=self.tager, task=tfdf.keras.Task.CLASSIFICATION)
            test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(X_test, label=self.tager, task=tfdf.keras.Task.CLASSIFICATION)

            # instantiate the model
            model_rf = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.CLASSIFICATION)

            # optional step - add evaluation metrics
            model_rf.compile(metrics=[ "acc"])

            # fit the model
            model_rf.fit(x=train_ds) 

        except:
            print('too few classes')

        return model_rf, test_ds

    def evaluate_model(self, model_rf, test_ds):
        """
        evaluate the model 

        :param dataset: <h5> the name model
        :param test_ds: <tensor slice> the test set
        :return: <float> returns the evaluation of the model
        """

        evaluation = model_rf.evaluate(test_ds, return_dict=True)

        return evaluation

    
    def predict_data(self, model_rf, new_data):
        """
        predict new dtaa

        :param dataset: <h5> the name model
        :param new_data: <tensor slice> new data to predict
        :return: <float> returns the predictions
        """
        model_rf.predict(new_data)
        
        return

if __name__ == '__main__':

    processor = DataProcessor("data/german/german.data")
    df = processor.load_data()    

    classifier = TfdfClassifier(df,21)

    model_rf, test_ds = classifier.train_classifier()
    evaluation = classifier.evaluate_model( model_rf, test_ds)

