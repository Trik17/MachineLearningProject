# dataframe management
import pandas as pd
# numerical computation
import numpy as np

class Dataset:

    #static pseidoPrivate:
    #__prova = 5
    #static public:
    #prova = 5

    def __init__(self):
        # Reading the data
        self.data = pd.read_csv('responses.csv')
        #self.X = self.data.drop(columns=['Empathy'])
        #self.Y = self.data['Empathy']

#with this it is possible to run the code only if this class is runned
#(otherwise it is runned everytime that the class is imported)
#if __name__ == "__main__":

