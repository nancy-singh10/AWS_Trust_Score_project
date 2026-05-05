import boto3
import pandas as pd

s3 = boto3.client('s3')

bucket_name = "nancy-ai-project-data"
file_key = "aws_ready_dataset_large.csv"

obj = s3.get_object(Bucket=bucket_name, Key=file_key)

df = pd.read_csv(obj['Body'])

print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())
# Price segmentation
df['price_range'] = pd.cut(
    df['price'],
    bins=[0, 500, 2000, 5000, 10000],
    labels=['Low', 'Medium', 'High', 'Premium']
)

# Rating segmentation
df['rating_bucket'] = pd.cut(
    df['rating'],
    bins=[0, 2, 3.5, 5],
    labels=['Poor', 'Average', 'Good']
)
print(df['trust_label'].value_counts())
print(df.groupby('maincateg')['trust_score'].mean())
print(df.groupby('price_range')['trust_score'].mean())
df.to_csv("final_cleaned_dataset.csv", index=False)
df.to_parquet("final_cleaned_dataset.parquet", index=False)
import boto3

s3 = boto3.client('s3')

s3.upload_file(
    "final_cleaned_dataset.csv",
    "nancy-ai-project-data",
    "cleaned/final_cleaned_dataset.csv"
)
s3.upload_file(
    "final_cleaned_dataset.parquet",
    "nancy-ai-project-data",
    "cleaned/final_cleaned_dataset.parquet"
)