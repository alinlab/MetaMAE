import math


def get_split_df(df, train=True, train_ratio=0.9):
    if train:
        df = df.head(math.ceil(train_ratio * len(df)))
    else:
        df = df.tail(math.floor((1 - train_ratio) * len(df)))
    return df