import pandas as pd

# Load datasets
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

# Add labels
fake['label'] = 0   # Fake
true['label'] = 1   # Real

# Combine
data = pd.concat([fake, true], axis=0)

# Keep only required columns
data = data[['text', 'label']]

# Shuffle dataset
data = data.sample(frac=1).reset_index(drop=True)

# Save final dataset
data.to_csv("dataset/fake_news.csv", index=False)

print("✅ fake_news.csv created successfully")
