import pandas as pd
df = pd.read_json ('dialog.json')
df.to_csv ('korean_dialog.csv', index = None)

print(df)