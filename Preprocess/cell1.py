"""Environment setup and data loading module.

This module handles the initial setup for the anti-money laundering (AML)
detection pipeline. It configures the environment, imports necessary libraries,
and loads the three primary datasets: transactions, alerts, and predictions.

The module performs the following tasks:
    1. Imports all required libraries (PyTorch, scikit-learn, XGBoost, etc.)
    2. Suppresses warnings and clears GPU cache
    3. Loads CSV files for transactions, alerts, and predictions
    4. Validates file existence and reports data shapes
    5. Sets the base date for temporal feature engineering

Attributes:
    PATH_TXN (str): Path to the transaction CSV file.
    PATH_ALERT (str): Path to the alert CSV file.
    PATH_PREDICT (str): Path to the prediction targets CSV file.
    base_date (pd.Timestamp): Reference date (2025-01-01) for temporal features.
    txn (pd.DataFrame): Transaction records with columns like from_acct, to_acct, txn_amt.
    alert (pd.DataFrame): Known alert/fraud accounts.
    pred (pd.DataFrame): Accounts to predict for submission.

Raises:
    FileNotFoundError: If any of the required CSV files do not exist.

Example:
    This module is executed as part of the preprocessing pipeline::

        $ python Preprocess/cell1.py
        開始讀取資料...
        資料讀取完成: txn=(1000000, 8), alert=(500, 2), pred=(200, 1)
        Cell 1 完成！
"""

import sys, os, gc, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
torch.cuda.empty_cache()
gc.collect()

# 檔案路徑（請依你的路徑調整）
PATH_TXN = "./acct_transaction.csv"
PATH_ALERT = "./acct_alert.csv"
PATH_PREDICT = "./acct_predict.csv"

for p in [PATH_TXN, PATH_ALERT, PATH_PREDICT]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"找不到檔案: {p}")

print("開始讀取資料...")
txn = pd.read_csv(PATH_TXN)
alert = pd.read_csv(PATH_ALERT)
pred = pd.read_csv(PATH_PREDICT)
print(f"資料讀取完成: txn={txn.shape}, alert={alert.shape}, pred={pred.shape}")

# 時間基準
base_date = pd.to_datetime('2025-01-01')
print("Cell 1 完成！")