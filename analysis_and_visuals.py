import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# -----------------
# Parameters
# -----------------
COMPANIES = ["AAPL", "MSFT", "AMZN", "GOOGL", "META",
             "JNJ", "JPM", "XOM", "PG", "NVDA"]
TECH = ["AAPL","MSFT","AMZN","GOOGL","META","NVDA"]
BANKS = ["JPM"]
MARKET = "^GSPC"
PERIOD = "2y"
RESAMPLE = "W"  # weekly returns

# -----------------
# Download price data
# -----------------
tickers = COMPANIES + [MARKET]
data = yf.download(tickers, period=PERIOD, interval="1d")["Close"]
returns = data.pct_change().resample(RESAMPLE).sum().dropna()
market_ret = returns[MARKET]

# -----------------
# Get last available multiples per company
# -----------------
last_multiples = {}
for ticker in COMPANIES:
    for mult in ['P/E','P/B','EV/EBITDA','P/S']:
        # Take last non-NaN value
        last_value = multiples_df[(ticker, mult)].dropna().iloc[-1] if not multiples_df[(ticker, mult)].dropna().empty else np.nan
        last_multiples[(ticker, mult)] = last_value
        
# -----------------
# Conditional CAPM (Beta+ / Beta-)
# -----------------
capm_results = {}
for ticker in COMPANIES:
    y = returns[ticker]
    up_mask = market_ret > 0
    down_mask = market_ret <= 0

    X_up = sm.add_constant(market_ret[up_mask])
    model_up = sm.OLS(y[up_mask], X_up).fit()
    alpha_up, beta_up = model_up.params

    X_down = sm.add_constant(market_ret[down_mask])
    model_down = sm.OLS(y[down_mask], X_down).fit()
    alpha_down, beta_down = model_down.params

    capm_results[ticker] = {
        "Beta+": beta_up,
        "Alpha+": alpha_up,
        "Beta-": beta_down,
        "Alpha-": alpha_down
    }

capm_df = pd.DataFrame(capm_results).T

# -----------------
# Fetch quarterly multiples
# -----------------
multiples = {}
for ticker in COMPANIES:
    stock = yf.Ticker(ticker)
    try:
        fin = stock.quarterly_financials.T
        bal = stock.quarterly_balance_sheet.T
    except Exception as e:
        print(f"Skipping {ticker}: {e}")
        continue

    # Safe access
    net_income = fin.get('Net Income', pd.Series(np.nan, index=fin.index))
    total_revenue = fin.get('Total Revenue', pd.Series(np.nan, index=fin.index))
    total_assets = bal.get('Total Assets', pd.Series(np.nan, index=bal.index))
    total_equity = bal.get('Total Stockholder Equity', pd.Series(np.nan, index=bal.index))
    ebit = fin.get('EBIT', pd.Series(np.nan, index=fin.index))
    ebitda = fin.get('EBITDA', pd.Series(np.nan, index=fin.index))
    market_cap_val = stock.info.get('marketCap', np.nan)

    # Default multiples
    pe = total_revenue / net_income.replace(0, np.nan)
    pb = total_assets / total_equity.replace(0, np.nan)
    ev_ebitda = ebit / ebitda.replace(0, np.nan)
    ps = total_revenue / market_cap_val

    # Substitute missing P/B
    if ticker in TECH and pb.isna().all():
        pb = ps  # tech: use P/S
    elif ticker in BANKS and pb.isna().all():
        ev = market_cap_val + stock.info.get('totalDebt',0) - stock.info.get('cash',0)
        pb = ev / total_assets  # bank: use EV / Assets

    multiples[ticker] = pd.DataFrame({
        'P/E': pe,
        'P/B': pb,
        'EV/EBITDA': ev_ebitda,
        'P/S': ps
    })

# Combine multiples
multiples_df = pd.concat(multiples, axis=1)
multiples_df.columns = pd.MultiIndex.from_product([COMPANIES, ['P/E','P/B','EV/EBITDA','P/S']])
multiples_df = multiples_df.ffill().resample(RESAMPLE).ffill()  # align to weekly returns

# -----------------
# Correlation matrix
# -----------------
corr_matrix = pd.DataFrame(index=COMPANIES, columns=['P/E','P/B','EV/EBITDA','P/S'], dtype=float)

for ticker in COMPANIES:
    for mult in ['P/E','P/B','EV/EBITDA','P/S']:
        combined = pd.concat([returns[ticker], multiples_df[(ticker, mult)]], axis=1, join='inner').dropna()
        if len(combined) > 1:
            corr_matrix.loc[ticker, mult] = combined.iloc[:,0].corr(combined.iloc[:,1])
        else:
            corr_matrix.loc[ticker, mult] = np.nan

# Heatmap plotting
def plot_heatmap(df, title):
    plt.figure(figsize=(10,6))
    sns.heatmap(df.astype(float), annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)
    plt.title(title)
    plt.yticks(rotation=0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_heatmap(corr_matrix, "Correlation: Weekly Stock Returns vs Multiples")

# -----------------
# Conditional CAPM scatter plots
# -----------------
for ticker in COMPANIES:
    y = returns[ticker]
    up_mask = market_ret > 0
    down_mask = market_ret <= 0

    plt.figure(figsize=(6,4))
    plt.scatter(market_ret[up_mask], y[up_mask], color='blue', alpha=0.5, s=20, label="Market Up")
    plt.scatter(market_ret[down_mask], y[down_mask], color='red', alpha=0.5, s=20, label="Market Down")

    alpha_up = capm_df.loc[ticker, "Alpha+"]
    beta_up = capm_df.loc[ticker, "Beta+"]
    alpha_down = capm_df.loc[ticker, "Alpha-"]
    beta_down = capm_df.loc[ticker, "Beta-"]

    x_vals_up = np.linspace(market_ret[up_mask].min(), market_ret[up_mask].max(), 100)
    x_vals_down = np.linspace(market_ret[down_mask].min(), market_ret[down_mask].max(), 100)

    plt.plot(x_vals_up, alpha_up + beta_up * x_vals_up, color='blue', linewidth=2)
    plt.plot(x_vals_down, alpha_down + beta_down * x_vals_down, color='red', linewidth=2)

    plt.axhline(0, color='black', linewidth=0.7)
    plt.axvline(0, color='black', linewidth=0.7)
    plt.title(f"Conditional Beta: {ticker} vs S&P500")
    plt.xlabel("Market Return (S&P500)")
    plt.ylabel("Stock Return")
    plt.legend()
    plt.show()

## -----------------
# Correlation: Beta+ / Beta- vs last multiples (ignore NaNs)
# -----------------
beta_corr_matrix = pd.DataFrame(index=["Beta+", "Beta-"], columns=['P/E','P/B','EV/EBITDA','P/S'], dtype=float)

for mult in ['P/E','P/B','EV/EBITDA','P/S']:
    last_vals = np.array([last_multiples[(ticker, mult)] for ticker in COMPANIES], dtype=float)
    beta_plus_vals = np.array(capm_df["Beta+"].values, dtype=float)
    beta_minus_vals = np.array(capm_df["Beta-"].values, dtype=float)

    # Remove NaNs
    valid_idx_plus = ~np.isnan(last_vals) & ~np.isnan(beta_plus_vals)
    valid_idx_minus = ~np.isnan(last_vals) & ~np.isnan(beta_minus_vals)

    if valid_idx_plus.any():
        beta_corr_matrix.loc["Beta+", mult] = np.corrcoef(beta_plus_vals[valid_idx_plus], last_vals[valid_idx_plus])[0,1]
    else:
        beta_corr_matrix.loc["Beta+", mult] = np.nan

    if valid_idx_minus.any():
        beta_corr_matrix.loc["Beta-", mult] = np.corrcoef(beta_minus_vals[valid_idx_minus], last_vals[valid_idx_minus])[0,1]
    else:
        beta_corr_matrix.loc["Beta-", mult] = np.nan

# Plot heatmap
plt.figure(figsize=(8,3))
sns.heatmap(beta_corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)
plt.title("Correlation: Beta+ / Beta- vs Last Multiples (NaNs ignored)")
plt.yticks(rotation=0)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------
# Key multiples table
# -----------------
key_multiples_table = pd.DataFrame(index=COMPANIES, columns=['P/E','P/B','EV/EBITDA','P/S'], dtype=float)
for ticker in COMPANIES:
    for mult in ['P/E','P/B','EV/EBITDA','P/S']:
        key_multiples_table.loc[ticker, mult] = last_multiples[(ticker, mult)]

print("\n=== Key Multiples per Company ===")
print(key_multiples_table)

# -----------------
# Combined CAPM + multiples summary
# -----------------
combined_summary = capm_df.copy()
for mult in ['P/E','P/B','EV/EBITDA','P/S']:
    combined_summary[mult] = [last_multiples[(ticker, mult)] for ticker in COMPANIES]

print("\n=== Combined CAPM + Multiples Summary ===")
print(combined_summary)
