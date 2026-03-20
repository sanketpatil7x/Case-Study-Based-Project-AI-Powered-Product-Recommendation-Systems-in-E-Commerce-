import pandas as pd

columns = ["user_id", "item_id", "rating", "timestamp"]

ratings = pd.read_csv(
    "../data/raw/u.data",
    sep="\t",
    names=columns
)