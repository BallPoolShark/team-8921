"""Advanced feature engineering module for AML detection.

This module generates a comprehensive set of features for anti-money laundering
detection, including:
    1. Bidirectional aggregation features (from/to accounts)
    2. High-risk country and currency indicators
    3. One-way transaction patterns
    4. Temporal patterns (safe/unsafe hours)
    5. Channel risk scores
    6. Cross-bank transaction indicators
    7. Small transaction ratio features

The features are designed to capture suspicious transaction patterns such as
structuring, layering, and integration commonly associated with money laundering.

Attributes:
    feat (pd.DataFrame): Complete feature matrix with all engineered features
        and labels for training.

Example:
    This module is executed as part of the preprocessing pipeline::

        $ python Preprocess/cell3.py
        開始產生高分特徵...
        基本聚合完成
        高風險國家/貨幣特徵完成
        單向交易特徵完成
        高分特徵完成！Shape: (50000, 150)
        Cell 3 成功！
"""

print("開始產生高分特徵...")

import pandas as pd
import numpy as np

# === 定義 build_side_features_fast ===
def build_side_features_fast(df, prefix, group_col):
    """Build aggregated features for accounts from transaction data.

    This function creates statistical and categorical features by aggregating
    transaction data grouped by account. It generates features such as
    transaction counts, amounts (sum/mean/std), temporal patterns, and
    categorical distributions.

    Args:
        df (pd.DataFrame): Transaction DataFrame with columns like txn_amt,
            txn_hour, channel_type, from_acct, to_acct.
        prefix (str): Prefix for feature column names (e.g., 'from' or 'to').
        group_col (str): Column name to group by (e.g., 'from_acct' or 'to_acct').

    Returns:
        pd.DataFrame: Aggregated features with columns:
            - acct: Account identifier
            - {prefix}_txn_cnt: Number of transactions
            - {prefix}_amt_sum/mean/std: Amount statistics
            - {prefix}_hour_mean: Average transaction hour
            - {prefix}_ch_nuniq: Number of unique channels
            - {prefix}_{category}_*: One-hot encoded categorical features

    Note:
        The function handles missing currency_type columns gracefully and
        performs one-hot encoding for categorical features like is_self_txn,
        channel_type, and account types.

    Example:
        >>> feat_from = build_side_features_fast(txn, 'from', 'from_acct')
        >>> feat_from.head()
           acct  from_txn_cnt  from_amt_sum  from_amt_mean  ...
        0  A001           100     1000000.0        10000.0  ...
    """
    agg_dict = {
        'txn_amt': ['count', 'sum', 'mean', 'std'],
        'txn_hour': 'mean',
        'channel_type': 'nunique'
    }
    if 'currency_type' in df.columns:
        agg_dict['currency_type'] = 'nunique'
    
    g = df.groupby(group_col, observed=True)
    base = g.agg(agg_dict)
    base.columns = [f"{prefix}_{c}_{a}" if a != '<lambda>' else f"{prefix}_{c}" for c, a in base.columns]
    base = base.reset_index()
    base.rename(columns={
        f"{prefix}_txn_amt_count": f"{prefix}_txn_cnt",
        f"{prefix}_txn_amt_sum": f"{prefix}_amt_sum",
        f"{prefix}_txn_amt_mean": f"{prefix}_amt_mean",
        f"{prefix}_txn_amt_std": f"{prefix}_amt_std",
        f"{prefix}_txn_hour_mean": f"{prefix}_hour_mean",
        f"{prefix}_channel_type_nunique": f"{prefix}_ch_nuniq",
        f"{prefix}_currency_type_nunique": f"{prefix}_curr_nuniq"
    }, inplace=True)
    
    cat_cols = ['is_self_txn', 'from_acct_type', 'to_acct_type', 'channel_type']
    dummies = []
    for c in cat_cols:
        if c in df.columns:
            dum = pd.get_dummies(df[[group_col, c]], columns=[c], prefix=f"{prefix}_{c}")
            dum = dum.groupby(group_col, observed=True).sum()
            dummies.append(dum)
    if dummies:
        cat_feat = pd.concat(dummies, axis=1).reset_index()
        base = base.merge(cat_feat, on=group_col, how='left')
    
    base = base.fillna(0)
    base = base.rename(columns={group_col: 'acct'})
    return base

# === 雙向聚合 ===
feat_from = build_side_features_fast(txn_allowed, 'from', 'from_acct')
feat_to = build_side_features_fast(txn_allowed, 'to', 'to_acct')
feat = pd.merge(feat_from, feat_to, on='acct', how='outer').fillna(0)
feat['net_amt'] = feat['from_amt_sum'] - feat['to_amt_sum']
feat['io_ratio'] = feat['to_amt_sum'] / (feat['from_amt_sum'] + 1e-3)
print("基本聚合完成")

# === 高風險國家/貨幣 ===
high_risk_currencies = ['CNY', 'HKD', 'THB', 'JPY', 'ZAR', 'MYR', 'PHP', 'KHR', 'MMK']
currency_to_country_risk = {
    'CNY': 8.5, 'HKD': 7.8, 'THB': 7.2, 'JPY': 6.5, 'ZAR': 6.0,
    'MYR': 7.0, 'PHP': 6.8, 'KHR': 8.2, 'MMK': 9.0,
    'USD': 4.0, 'EUR': 2.5, 'TWD': 1.0, 'NTD': 1.0
}
if 'currency_type' in txn_allowed.columns:
    txn_allowed['is_high_risk_curr'] = txn_allowed['currency_type'].isin(high_risk_currencies).astype(int)
    txn_allowed['curr_country_risk'] = txn_allowed['currency_type'].map(currency_to_country_risk).fillna(1.0)
    
    g_from = txn_allowed.groupby('from_acct').agg({
        'is_high_risk_curr': ['mean', 'sum', 'nunique'],
        'curr_country_risk': ['mean', 'sum']
    }).round(4)
    g_from.columns = ['_'.join(c) if c[1] != 'nunique' else c[0] + '_nuniq' for c in g_from.columns]
    g_from = g_from.reset_index().rename(columns={'from_acct': 'acct'})
    
    g_to = txn_allowed.groupby('to_acct').agg({
        'is_high_risk_curr': ['mean', 'sum', 'nunique'],
        'curr_country_risk': ['mean', 'sum']
    }).round(4)
    g_to.columns = ['_'.join(c) if c[1] != 'nunique' else c[0] + '_nuniq' for c in g_to.columns]
    g_to = g_to.reset_index().rename(columns={'to_acct': 'acct'})
    
    feat = feat.merge(g_from, on='acct', how='left').merge(g_to, on='acct', how='left').fillna(0)
    print("高風險國家/貨幣特徵完成")

# === 單向交易 ===
txn_clean = txn_allowed.dropna(subset=['from_acct', 'to_acct']).copy()
partners_from = txn_clean.groupby('from_acct')['to_acct'].nunique().reset_index(name='from_total_partners')
partners_to = txn_clean.groupby('to_acct')['from_acct'].nunique().reset_index(name='to_total_partners')
feat = feat.merge(partners_from, left_on='acct', right_on='from_acct', how='left')
feat = feat.merge(partners_to, left_on='acct', right_on='to_acct', how='left')

all_pairs = txn_clean[['from_acct', 'to_acct']].drop_duplicates()
pairs_set = set()
for row in all_pairs.itertuples(index=False):
    pair = tuple(sorted([row.from_acct, row.to_acct]))
    pairs_set.add(pair)

mutual_pairs = {p for p in pairs_set if (p[1], p[0]) in pairs_set}
txn_clean['is_one_way'] = txn_clean.apply(lambda r: tuple(sorted([r.from_acct, r.to_acct])) not in mutual_pairs, axis=1)

oneway_from = txn_clean[txn_clean['is_one_way']].groupby('from_acct')['to_acct'].nunique().reset_index(name='from_oneway_partners')
oneway_to = txn_clean[txn_clean['is_one_way']].groupby('to_acct')['from_acct'].nunique().reset_index(name='to_oneway_partners')
feat = feat.merge(oneway_from, left_on='acct', right_on='from_acct', how='left')
feat = feat.merge(oneway_to, left_on='acct', right_on='to_acct', how='left')

feat['from_oneway_ratio'] = feat['from_oneway_partners'] / feat['from_total_partners'].clip(1)
feat['to_oneway_ratio'] = feat['to_oneway_partners'] / feat['to_total_partners'].clip(1)
feat.fillna(0, inplace=True)
print("單向交易特徵完成")

# === PPT 三特徵：is_safe_hour ===
txn_allowed['is_safe_hour'] = (~txn_allowed['txn_hour'].between(9, 18)).astype(int)
g_from = txn_allowed.groupby('from_acct')['is_safe_hour'].agg(['mean', 'sum']).reset_index()
g_from = g_from.rename(columns={'from_acct': 'acct', 'mean': 'from_is_safe_hour_ratio', 'sum': 'from_is_safe_hour_cnt'})
g_to = txn_allowed.groupby('to_acct')['is_safe_hour'].agg(['mean', 'sum']).reset_index()
g_to = g_to.rename(columns={'to_acct': 'acct', 'mean': 'to_is_safe_hour_ratio', 'sum': 'to_is_safe_hour_cnt'})
feat = feat.merge(g_from, on='acct', how='left').merge(g_to, on='acct', how='left').fillna(0)
print("is_safe_hour 特徵完成")

# === channel_risk_score ===
channel_risk_map = {'ATM(1)': 2.0, '行動銀行(3)': 1.5, '線上銀行(4)': 1.0, '櫃台(2)': 0.5, 'UNK': 0.5}
txn_allowed['channel_risk_score'] = txn_allowed['channel_type'].map(channel_risk_map).fillna(0.5)
g_from = txn_allowed.groupby('from_acct')['channel_risk_score'].agg(['mean', 'sum']).reset_index()
g_from = g_from.rename(columns={'from_acct': 'acct', 'mean': 'from_channel_risk_mean', 'sum': 'from_channel_risk_sum'})
g_to = txn_allowed.groupby('to_acct')['channel_risk_score'].agg(['mean', 'sum']).reset_index()
g_to = g_to.rename(columns={'to_acct': 'acct', 'mean': 'to_channel_risk_mean', 'sum': 'to_channel_risk_sum'})
feat = feat.merge(g_from, on='acct', how='left').merge(g_to, on='acct', how='left').fillna(0)
print("channel_risk_score 特徵完成")

# === is_cross_bank ===
txn_allowed['is_cross_bank'] = (txn_allowed['from_acct_type'] != txn_allowed['to_acct_type']).astype(int)
g_from = txn_allowed.groupby('from_acct')['is_cross_bank'].agg(['mean', 'sum']).reset_index()
g_from = g_from.rename(columns={'from_acct': 'acct', 'mean': 'from_is_cross_bank_ratio', 'sum': 'from_is_cross_bank_cnt'})
g_to = txn_allowed.groupby('to_acct')['is_cross_bank'].agg(['mean', 'sum']).reset_index()
g_to = g_to.rename(columns={'to_acct': 'acct', 'mean': 'to_is_cross_bank_ratio', 'sum': 'to_is_cross_bank_cnt'})
feat = feat.merge(g_from, on='acct', how='left').merge(g_to, on='acct', how='left').fillna(0)
print("is_cross_bank 特徵完成")

# === 清理多餘欄 ===
cols_to_drop = [col for col in feat.columns if col in ['from_acct', 'to_acct'] or ('_x' in col and 'from_acct' in col)]
feat.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# === 標籤 ===
candidates = pd.concat([txn_allowed['from_acct'], txn_allowed['to_acct']]).unique()
labels = pd.DataFrame({'acct': candidates})
labels = labels.merge(alert[['acct']], on='acct', how='left', indicator=True)
labels['label'] = (labels['_merge'] == 'both').astype(int)
labels = labels[['acct', 'label']]
feat = feat.merge(labels, on='acct', how='left').fillna({'label': 0})
feat['label'] = feat['label'].astype(int)

print(f"高分特徵完成！Shape: {feat.shape}")
print(f"Label: 0={feat['label'].value_counts().get(0,0)}, 1={feat['label'].value_counts().get(1,0)}")
gc.collect()
print("Cell 3 成功！可跑 Cell 4-6")




# 在 Cell 3 最後，加這兩行（不影響其他）
txn_allowed['is_small_txn'] = (txn_allowed['txn_amt'] < 1000).astype(int)
small_from = txn_allowed.groupby('from_acct')['is_small_txn'].agg(['mean', 'sum']).reset_index()
small_from = small_from.rename(columns={'from_acct': 'acct', 'mean': 'from_small_ratio', 'sum': 'from_small_cnt'})
small_to = txn_allowed.groupby('to_acct')['is_small_txn'].agg(['mean', 'sum']).reset_index()
small_to = small_to.rename(columns={'to_acct': 'acct', 'mean': 'to_small_ratio', 'sum': 'to_small_cnt'})
feat = feat.merge(small_from, on='acct', how='left').merge(small_to, on='acct', how='left').fillna(0)
print("小額比例特徵完成")