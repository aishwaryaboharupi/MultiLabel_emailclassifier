import pandas as pd
from sklearn.model_selection import train_test_split
import config

class DataHandler:
    def __init__(self):
        self.df = None

    def load_process_data(self):
        # 1. Load data from the config path (purchase.csv)
        self.df = pd.read_csv(config.DATA_PATH)
       
        # 2. Rule: Ignore Type 1 (Separation of Concerns)
        if 'Type 1' in self.df.columns:
            self.df = self.df.drop(columns=['Type 1'])
           
        # 3. Design Choice 1: Create Chained Labels (y2, y23, y234)
        # We handle NaN values just in case they exist so the strings combine properly
        self.df = self.df.fillna("Unknown")
       
        self.df['y2'] = self.df['Type 2']
        self.df['y23'] = self.df['Type 2'] + " + " + self.df['Type 3']
        self.df['y234'] = self.df['Type 2'] + " + " + self.df['Type 3'] + " + " + self.df['Type 4']
       
        return self.df

    def get_splits(self, target_name):
        # UPDATED: Changed 'Content' to 'Ticket Summary' to match your CSV
        X = self.df['Ticket Summary']
        y = self.df[target_name]
        return train_test_split(X, y, test_size=0.2, random_state=42)