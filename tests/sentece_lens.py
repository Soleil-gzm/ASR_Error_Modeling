import pandas as pd
df = pd.read_csv('work/test_gpt2_sample_10_pt/intermediate/all_sentences.csv')
lens = df['sentence'].str.len()
print(lens.describe(percentiles=[0.5, 0.9, 0.95, 0.99]))