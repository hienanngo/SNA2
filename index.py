import kagglehub
from kagglehub import KaggleDatasetAdapter

# IMPORTANT: Use the exact file name shown on Kaggle
file_path = "reviews (3).csv"

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "hinanng/washington-d-c-airbnb-reviews",
    file_path
)

print(df.head())
