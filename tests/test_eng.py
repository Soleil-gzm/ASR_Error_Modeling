import pandas as pd
print("gpt:")
df_gpt = pd.read_csv('work/test_Qwen_sample_10_pt/outputs/word_nll_details_sample_10.csv')
print(df_gpt[df_gpt['token'].str.contains('eight', na=False)])
print("\nqwen:")
df_qwen = pd.read_csv('work/test_gpt2_sample_10_pt/outputs/sample_10_analysis/report/word_level_aggregated_sample_10.csv')
print(df_qwen[df_qwen['word'].str.contains('eight', na=False)])