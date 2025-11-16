"""Data preprocessing module with currency conversion and data leakage prevention.

This module performs critical preprocessing steps for transaction data:
    1. Currency conversion to NTD (New Taiwan Dollar)
    2. Temporal feature extraction (transaction hour)
    3. Data leakage prevention to avoid using future information
    4. Missing value imputation for categorical features

The data leakage prevention ensures that transactions occurring after an
alert event date are excluded from the training set to maintain temporal
validity.

Attributes:
    txn_allowed (pd.DataFrame): Filtered transaction data after leakage prevention.

Example:
    This module is executed as part of the preprocessing pipeline::

        $ python Preprocess/cell2.py
        防洩漏後: 950000 / 1000000
        Cell 2 完成！
"""

def ConvertToNTD(df):
    """Convert transaction amounts to NTD using predefined exchange rates.

    This function standardizes all transaction amounts to New Taiwan Dollar (NTD)
    by applying currency-specific exchange rates. It handles multiple currency
    types including USD, EUR, JPY, CNY, and others.

    Args:
        df (pd.DataFrame): DataFrame containing transaction data with columns
            'txn_amt' and either 'currency_type' or 'currency'.

    Returns:
        pd.DataFrame: The input DataFrame with an additional column 'txn_amt_ntd'
            containing the converted amounts in NTD.

    Note:
        Exchange rates are hardcoded based on approximate 2025 values:
        - USD: 31.5, EUR: 34.5, JPY: 0.215, CNY: 4.35
        - If currency is not found in the mapping, defaults to rate 1.0
        - If no currency column exists, txn_amt is used directly

    Example:
        >>> df = pd.DataFrame({'txn_amt': [100, 200], 'currency_type': ['USD', 'EUR']})
        >>> df_converted = ConvertToNTD(df)
        >>> df_converted['txn_amt_ntd']
        0    3150.0
        1    6900.0
    """
    rates = {'NTD':1.0,'TWD':1.0,'USD':31.5,'EUR':34.5,'JPY':0.215,'CNY':4.35,
             'HKD':4.05,'AUD':20.5,'GBP':39.5,'CAD':23.0,'SGD':23.5,
             'NZD':18.8,'CHF':35.0,'THB':0.88,'ZAR':1.65,'SEK':2.95,'MXN':1.85}
    col = 'currency_type' if 'currency_type' in df.columns else 'currency' if 'currency' in df.columns else None
    if col:
        df['txn_amt_ntd'] = df.apply(lambda r: r['txn_amt'] * rates.get(r[col], 1.0), axis=1)
    else:
        df['txn_amt_ntd'] = df['txn_amt']
    return df

txn = ConvertToNTD(txn)
txn['txn_amt'] = txn['txn_amt_ntd']

# 時間處理
txn['txn_hour'] = pd.to_datetime(txn['txn_time'], errors='coerce').dt.hour.fillna(12).astype(int)
txn['txn_date_converted'] = base_date + pd.to_timedelta(txn['txn_date'], unit='D')

# 防洩漏
if 'event_date' in alert.columns:
    alert['event_date_converted'] = base_date + pd.to_timedelta(alert['event_date'], unit='D')
    alert_map = alert.set_index('acct')['event_date_converted'].to_dict()
    txn['from_event'] = txn['from_acct'].map(alert_map)
    txn['to_event'] = txn['to_acct'].map(alert_map)
    mask = (~txn['from_event'].notna()) | (txn['txn_date_converted'] < txn['from_event'])
    mask &= (~txn['to_event'].notna()) | (txn['txn_date_converted'] < txn['to_event'])
    txn_allowed = txn[mask].copy()
    print(f"防洩漏後: {len(txn_allowed):,} / {len(txn):,}")
else:
    txn_allowed = txn.copy()
    print("無 event_date，跳過防洩漏")

# 填補缺失
for col in ['is_self_txn','channel_type','from_acct_type','to_acct_type']:
    if col not in txn_allowed.columns:
        txn_allowed[col] = 'UNK' if col in ['is_self_txn','channel_type'] else '00'
    txn_allowed[col] = txn_allowed[col].astype('category')

print("Cell 2 完成！")
gc.collect()