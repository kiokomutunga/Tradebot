# %%
# XAU/USD Short-Term Ready-to-Train Notebook (script-style)
# Save this file as `xauusd_short_term_trainer.py` or paste into a Jupyter cell file.
# Requirements: see requirements.txt (includes yfinance, pandas, ta, xgboost, scikit-learn, joblib, matplotlib)

# %% [markdown]
# ## 1. Imports & Config

# %%
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import timedelta

# indicators
import ta

# modeling
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.calibration import CalibratedClassifierCV

# persistence
import joblib

# %% [markdown]
# ## 2. Configuration — paths and hyperparameters
# Edit these if needed.

# %%
DATA_CSV = "gold_data.csv"     # your downloaded file (hourly candles)
MODEL_OUT = "xgb_xau_model.joblib"
SIGNALS_OUT = "predicted_signals.csv"

# labeling
FUTURE_H = 6       # prediction horizon in hours (e.g., 6 hours ahead)
RET_THRESH = 0.003 # 0.3% return threshold

# modeling
FEATURES = None    # will be set after features() call
PROB_THRESHOLD = 0.70  # only signal trades with probability >= this

# training split
TRAIN_RATIO = 0.75

# %% [markdown]
# ## 3. Data ingestion (load CSV or attempt yfinance fallback)

# %%
import yfinance as yf

def load_data(path=DATA_CSV):
    if os.path.exists(path):
        print(f"Loading data from {path}")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        # Normalize column names (common yfinance format sometimes capitalized)
        df.columns = [c.lower() for c in df.columns]
        # expect columns: open, high, low, close, volume (lowercase)
        if 'close' not in df.columns and ('adj close' in df.columns):
            df['close'] = df['adj close']
    else:
        print("Local CSV not found — trying yfinance fallback (GLD, last 730d hourly)")
        df = yf.download("GLD", period="730d", interval="1h")
        df.columns = [c.lower() + ("" if isinstance(c, str) else "") for c in df.columns]
    # Standardize to expected names
    rename_map = {}
    for col in df.columns:
        if col.lower().startswith('open'):
            rename_map[col] = 'open'
        if col.lower().startswith('high'):
            rename_map[col] = 'high'
        if col.lower().startswith('low'):
            rename_map[col] = 'low'
        if col.lower().startswith('close') or col.lower().startswith('adj close'):
            rename_map[col] = 'close'
        if col.lower().startswith('volume'):
            rename_map[col] = 'volume'
    df = df.rename(columns=rename_map)
    df = df[['open','high','low','close'] + ([ 'volume'] if 'volume' in df.columns else [])]
    df = df.sort_index()
    # Ensure hourly frequency index (fill small gaps)
    df = df[~df.index.duplicated(keep='first')]
    df = df.asfreq('1H')
    df = df.fillna(method='ffill')
    return df


_df = load_data()
print('Loaded rows:', len(_df))
print(_df.head())

# %% [markdown]
# ## 4. Feature engineering function

# %%

def make_features(df):
    df = df.copy()
    # Basic returns
    df['close'] = df['close'].astype(float)
    df['return_1'] = df['close'].pct_change()
    df['logret_1'] = np.log(df['close']).diff()

    # Moving averages
    df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
    df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
    df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
    df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()

    # Momentum
    df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()

    # Volatility
    df['atr_14'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['volatility_20'] = df['return_1'].rolling(20).std()

    # Bollinger band related
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_mid'] = bb.bollinger_mavg()
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    # normalized position in the band
    df['bb_pct'] = (df['close'] - df['bb_mid']) / (df['bb_high'] - df['bb_low'])

    # Price action
    df['body'] = (df['close'] - df['open']).abs()
    df['upper_wick'] = df['high'] - df[['close','open']].max(axis=1)
    df['lower_wick'] = df[['close','open']].min(axis=1) - df['low']

    # Recent highs/lows (structure)
    df['hh_24'] = df['high'].rolling(24).max()
    df['ll_24'] = df['low'].rolling(24).min()
    df['break_hh_1'] = (df['close'] > df['hh_24'].shift(1)).astype(int)
    df['break_ll_1'] = (df['close'] < df['ll_24'].shift(1)).astype(int)

    # Time features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek

    df = df.dropna()
    return df


df_fe = make_features(_df)
print('After features rows:', len(df_fe))
print(df_fe.columns.tolist())

# %% [markdown]
# ## 5. Labeling (future horizon & threshold)

# %%

def label_future(df, H=FUTURE_H, T=RET_THRESH):
    df = df.copy()
    df['future_close'] = df['close'].shift(-H)
    df['future_return'] = df['future_close'] / df['close'] - 1
    # binary classification: 1 = take a long trade (future_return >= T), 0 = no-trade
    df['label'] = 0
    df.loc[df['future_return'] >= T, 'label'] = 1
    # optionally create sell label (short) by uncommenting below (ternary):
    # df.loc[df['future_return'] <= -T, 'label'] = -1
    df = df.dropna()
    return df


df_lab = label_future(df_fe, H=FUTURE_H, T=RET_THRESH)
print('After labeling rows:', len(df_lab))
print(df_lab[['close','future_close','future_return','label']].tail(6))

# %% [markdown]
# ## 6. Prepare features and train/test split (time-based)

# %%
# Choose feature columns programmatically (exclude target & future columns)
exclude_cols = set(['future_close','future_return','label'])
feature_cols = [c for c in df_lab.columns if c not in exclude_cols and df_lab[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
FEATURES = feature_cols
print('Using features:', FEATURES)

# train/test split (time series order)
train_n = int(len(df_lab)*TRAIN_RATIO)
train_df = df_lab.iloc[:train_n]
test_df = df_lab.iloc[train_n:]

X_train = train_df[FEATURES]
y_train = train_df['label']
X_test = test_df[FEATURES]
y_test = test_df['label']

print('Train rows:', len(X_train), 'Test rows:', len(X_test))

# %% [markdown]
# ## 7. Train baseline XGBoost model (with probability calibration)

# %%
model = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.03, subsample=0.8, use_label_encoder=False, eval_metric='logloss')
# calibrate probabilities with sigmoid (Platt) to get better prob estimates for thresholding
calibrator = CalibratedClassifierCV(base_estimator=model, cv=3, method='sigmoid')
calibrator.fit(X_train, y_train)

# save model
joblib.dump(calibrator, MODEL_OUT)
print('Saved calibrated model to', MODEL_OUT)

# %% [markdown]
# ## 8. Evaluation on test set

# %%
probs = calibrator.predict_proba(X_test)[:,1]
preds = (probs >= 0.5).astype(int)

print('Classification report (threshold 0.5):')
print(classification_report(y_test, preds))

# Confusion matrix
cm = confusion_matrix(y_test, preds)
print('Confusion matrix:\n', cm)

# Precision/Recall at high-confidence threshold
high_conf_mask = probs >= PROB_THRESHOLD
if high_conf_mask.sum() > 0:
    precision, recall, f1, _ = precision_recall_fscore_support(y_test[high_conf_mask], preds[high_conf_mask], average='binary', zero_division=0)
    print(f'High-confidence (p>={PROB_THRESHOLD}) rows: {high_conf_mask.sum()}')
    print(f'Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}')
else:
    print('No high-confidence predictions on test set with current PROB_THRESHOLD')

# %% [markdown]
# ## 9. Produce signals for the whole dataset (probabilities and high-confidence filter)

# %%
full_probs = calibrator.predict_proba(df_lab[FEATURES])[:,1]
df_signals = df_lab.copy()
df_signals['buy_prob'] = full_probs
df_signals['pred_label_05'] = (full_probs >= 0.5).astype(int)
df_signals['pred_label_high'] = (full_probs >= PROB_THRESHOLD).astype(int)

# extract only high-confidence buys as entry signals
signals = df_signals[df_signals['pred_label_high'] == 1][['close','future_return','buy_prob']]
signals = signals.copy()
signals['entry_time'] = signals.index

print('High-confidence signals found:', len(signals))
print(signals.tail(10))

# Save signals and full table
signals.to_csv(SIGNALS_OUT, index=True)
df_signals.to_csv('full_predictions.csv', index=True)
print('Saved high-confidence signals to', SIGNALS_OUT)

# %% [markdown]
# ## 10. Simple backtest of high-confidence signals (entry at close, exit at future_close)
# This is a simple deterministic backtest to get intuition about the signal performance.

# %%
bt = signals.copy()
bt['exit_time'] = bt.index + pd.Timedelta(hours=FUTURE_H)
bt['exit_price'] = df_lab['future_close'].loc[bt.index].values
bt['return_pct'] = bt['exit_price'] / bt['close'] - 1

# performance
wins = bt[bt['return_pct'] >= 0]
losses = bt[bt['return_pct'] < 0]
print('Backtest summary for high-confidence buys:')
print('Signals:', len(bt))
print('Wins:', len(wins), 'Losses:', len(losses))
print('Win rate:', len(wins)/len(bt) if len(bt) else None)
print('Average return (%):', bt['return_pct'].mean()*100 if len(bt) else None)
print('Cumulative return (%):', (bt['return_pct'] + 1).prod()*100 - 100 if len(bt) else None)

# %% [markdown]
# ## 11. Plots: probability distribution and equity curve of signals

# %%
plt.figure(figsize=(10,4))
plt.hist(df_signals['buy_prob'], bins=50)
plt.title('Predicted buy probability distribution')
plt.xlabel('Probability')
plt.ylabel('Count')
plt.show()

# equity curve of signals only (assume equal capital per trade)
if len(bt):
    returns = bt['return_pct']
    cum = (returns + 1).cumprod()
    plt.figure(figsize=(10,4))
    plt.plot(cum.index, cum.values)
    plt.title('Cumulative returns of high-confidence signals (per-signal sizing)')
    plt.ylabel('Cumulative multiplier')
    plt.show()

# %% [markdown]
# ## 12. Save artifacts and final notes

# %%
print('Artifacts saved:')
print(' - Model:', MODEL_OUT)
print(' - High-confidence signals:', SIGNALS_OUT)
print(' - All predictions: full_predictions.csv')

# Quick recommendation printed
print('\nRecommendations:')
print('- Tune FUTURE_H and RET_THRESH to balance frequency vs quality.')
print('- Increase PROB_THRESHOLD to get fewer but higher-precision signals.')
print('- Use walk-forward CV and a realistic backtester for production.')

# End of script
