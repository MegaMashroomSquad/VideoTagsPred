import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df[df['video_id'] != 'b4d70f82038d1d97f1b3ce2a493d12c8'].reset_index(drop=True)
    return df