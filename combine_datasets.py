import pandas as pd

# Load the datasets
true_df = pd.read_csv('True.csv')
fake_df = pd.read_csv('Fake.csv')

# Add a new column to label the data
true_df['label'] = 'REAL'
fake_df['label'] = 'FAKE'

# Combine the datasets
combined_df = pd.concat([true_df, fake_df], ignore_index=True)

# Shuffle the dataset (important for training)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to a new CSV file
combined_df.to_csv('fake_or_real_news.csv', index=False)

print("✅ Successfully combined True.csv and Fake.csv into fake_or_real_news.csv")
