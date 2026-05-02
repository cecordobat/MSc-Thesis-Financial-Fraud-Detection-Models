import pandas as pd
from src.utils.eda import create_mode_mapping

df = pd.read_parquet('data/processed/card_transaction.parquet')
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
print('rows', len(df))
print('null merchant_state', df['merchant_state'].isna().sum())
print('null zip', df['zip'].isna().sum())
m = create_mode_mapping(df, 'merchant_city', 'merchant_state')
print('mode mapping state size', len(m))
print('sample map', list(m.items())[:10])
miss = df[df['merchant_state'].isna() & df['merchant_city'].notna()].head(5)
print('missing examples', miss[['merchant_city','merchant_state','zip']].to_dict('records'))
