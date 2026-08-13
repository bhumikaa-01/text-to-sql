import pandas as pd

df = pd.read_csv("data/raw/olist_order_reviews_dataset.csv")

dupes = df[df["review_id"].duplicated(keep=False)]

print("Duplicate review_ids:", dupes["review_id"].nunique())

print(dupes[["review_id", "order_id"]].head(20))