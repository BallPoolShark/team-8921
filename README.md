# team-8921
2025玉山人工智慧挑戰賽 - 反洗錢偵測系統

## 專案簡介
本專案為反洗錢 (Anti-Money Laundering, AML) 偵測系統，使用機器學習和圖神經網路 (GNN) 技術來識別可疑的金融交易帳戶。

## 資料夾/檔案說明

### 主要程式檔案
- **[main.py](main.py)** - 主執行檔，依序執行資料預處理和模型訓練的完整流程

### 資料夾

#### [Preprocess/](Preprocess/)
資料預處理資料夾，包含四個處理步驟：

- **[cell1.py](Preprocess/cell1.py)** - 環境設定與資料載入
  - 匯入所需套件
  - 載入交易、警示、預測資料集
  - 設定基準日期

- **[cell2.py](Preprocess/cell2.py)** - 資料預處理與防洩漏
  - 貨幣轉換為新台幣 (NTD)
  - 時間特徵提取
  - 防止資料洩漏（排除未來資訊）
  - 缺失值填補

- **[cell3.py](Preprocess/cell3.py)** - 特徵工程
  - 建立帳戶特徵
  - 圖結構特徵（節點度、PageRank、中心性）
  - 交易模式特徵（金額統計、時間分佈）

- **[cell4.py](Preprocess/cell4.py)** - 標籤建立與資料分割
  - 建立訓練標籤
  - 合併特徵資料
  - 訓練/測試資料分割

#### [Model/](Model/)
模型訓練資料夾，包含兩個訓練步驟：

- **[cell5.py](Model/cell5.py)** - 圖神經網路 (GNN) 特徵提取
  - 使用 Graph Convolutional Network (GCN) 學習節點嵌入
  - 訓練無監督圖模型
  - 提取 64 維度的圖特徵

- **[cell6.py](Model/cell6.py)** - XGBoost 模型訓練與預測
  - 訓練 XGBoost 分類器
  - 處理類別不平衡問題
  - 模型評估與最佳閾值選擇
  - 生成預測結果與提交檔案

### 資料檔案

- **acct_transaction.csv** - 交易記錄資料（約 737 MB）
  - 包含帳戶間的交易金額、時間、類型等資訊

- **acct_alert.csv** - 警示帳戶資料
  - 已知的可疑/詐欺帳戶清單

- **acct_predict.csv** - 預測目標資料
  - 需要預測的帳戶清單



### 開發檔案

- **玉山3.0.ipynb** - Jupyter Notebook 開發檔案
  - 原始的探索性分析和模型開發筆記本

## 安裝說明

### 1. 安裝套件

使用 pip 安裝所需套件：

```bash
pip install -r requirements.txt
```

### 2. 確認資料檔案

確保以下三個 CSV 檔案位於與main.py同資料夾：
- acct_transaction.csv
- acct_alert.csv
- acct_predict.csv

## 使用方式

### 執行完整流程

直接執行 main.py 將依序運行所有預處理和訓練步驟：

```bash
python main.py
```

執行順序：
1. Preprocess/cell1.py - 資料載入
2. Preprocess/cell2.py - 資料預處理
3. Preprocess/cell3.py - 特徵工程
4. Preprocess/cell4.py - 標籤建立
5. Model/cell5.py - GNN 特徵提取
6. Model/cell6.py - XGBoost 訓練與預測

### 執行個別步驟

也可以單獨執行某個步驟的腳本：

```bash
python Preprocess/cell1.py
python Preprocess/cell2.py
# ... 以此類推
```

## 技術架構

### 特徵工程
- **圖結構特徵**: 節點度、PageRank、中心性指標
- **交易統計特徵**: 金額總和、平均、標準差、最大最小值
- **時間特徵**: 交易時段分佈、天數差異
- **GNN 嵌入**: 64 維圖神經網路特徵

### 模型架構
1. **圖神經網路 (GCN)**
   - 3 層卷積層
   - Batch Normalization
   - Dropout 正則化
   - 無監督邊重建損失

2. **XGBoost 分類器**
   - 處理類別不平衡
   - Precision-Recall 曲線最佳化
   - F1-score 導向的閾值選擇

## 系統需求

- Python 3.8+
- 建議使用 GPU (可選，會自動偵測)
- 記憶體: 至少 8GB RAM (資料集較大)

## 專案成果

本系統整合了傳統機器學習特徵工程與深度學習圖神經網路，能夠：
- 有效識別可疑交易帳戶
- 捕捉交易網路中的複雜模式
- 處理大規模交易資料（百萬級交易記錄）
- 防止資料洩漏，確保模型的泛化能力

## 授權

此專案為 2025 玉山人工智慧挑戰賽參賽作品。
