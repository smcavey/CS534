import pandas as pd

def main():
    df1 = pd.read_csv('board_data2.csv')
    df1 = df1.iloc[: , 1:]
    df2 = pd.read_csv('board_data3.csv')
    df2 = df2.iloc[: , 1:]
    df = pd.concat([df1, df2], axis=0)
    df.to_csv('tuples.csv')
main()