import pandas as pd
import os

class Data:

    def __init__(self):
        self.df = pd.read_csv(os.getcwd() + r'\data\train2022.csv')