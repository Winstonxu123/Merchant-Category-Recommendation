import pandas as pd
df = pd.read_csv('result/submission_train2_referenceDate8features_stack.csv', names=['card_id', 'target0'], skiprows=[0])
df2 = pd.read_csv('result/submission10.csv', names=['card_id', 'target1'], skiprows=[0])
df_base = pd.merge(df, df2, how='inner', on='card_id')
df_base['target'] = df_base['target0']*0.73839798 + df_base['target1']*0.26555364
df_base[['card_id', 'target']].to_csv('output.csv', index=False)