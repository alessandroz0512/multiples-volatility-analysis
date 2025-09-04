# Conditional CAPM and Fundamental Multiples Analysis

This project performs a **quantitative analysis of stock returns** for a set of major companies across tech, banking, and other sectors. The workflow integrates:

- Downloading historical price data and computing weekly returns  
- Fetching quarterly financial multiples (P/E, P/B, EV/EBITDA, P/S)  
- Conditional CAPM estimation (up-market vs down-market beta and alpha)  
- Correlation analysis between returns, betas, and financial multiples  
- Visualization of scatter plots, heatmaps, and trends

---

## Methodology

1. **Data Collection**
   - Stock price data for selected companies (AAPL, MSFT, AMZN, GOOGL, META, JNJ, JPM, XOM, PG, NVDA) and the S&P500 market index (^GSPC) is downloaded using `yfinance`.
   - Weekly returns are computed from daily closing prices.
   - Quarterly financials are extracted to compute valuation multiples: P/E, P/B, EV/EBITDA, and P/S.

2. **Conditional CAPM**
   - Returns are split into **up-market days** (market return > 0) and **down-market days** (market return ≤ 0).
   - Linear regression is applied to estimate conditional **alpha** and **beta** for each stock.
   - Results are stored in a structured dataframe for comparison.

3. **Multiples Correlation**
   - Historical multiples are aligned to weekly return data.
   - Correlations between weekly returns and multiples are computed for each company.
   - Heatmaps visualize the relationships.

4. **Beta vs Multiples**
   - Correlations between conditional betas (Beta+ / Beta-) and the last available multiples are computed.
   - This provides a proxy for how a company's valuation might be associated with its systematic risk in different market conditions.

5. **Visualization**
   - Scatter plots show conditional CAPM regressions for up- and down-market periods.
   - Heatmaps display correlations between returns, betas, and financial multiples.
   - Tables summarize key multiples per company and combined CAPM + multiples metrics.

---

## Analysis

- **Tech vs Non-Tech Dynamics:** Tech stocks often show higher beta in up-market periods, potentially indicating heightened sensitivity to market optimism, while defensive stocks like JNJ exhibit lower variability.
- **Asymmetric Beta:** Differences between Beta+ and Beta- suggest that some companies are more prone to market downturns than others, hinting at potential hedging or risk management strategies.
- **Valuation Multiples Speculation:** Observed correlations between Beta and multiples could imply that market participants price risk differently depending on underlying fundamentals. For instance, high P/E companies sometimes display stronger Beta+ responses, suggesting sensitivity to bullish sentiment.
- **Scatter Observations:** Visual patterns in conditional CAPM scatter plots suggest that market impact on returns is not strictly linear; clusters of outlier points may reflect earnings surprises or macroeconomic shocks.

---

## Results

1. **Conditional CAPM Estimates**
   - Beta+ and Beta- vary across sectors, revealing asymmetries in risk exposure.
   - Alpha values provide a rough gauge of abnormal returns under different market regimes.

2. **Multiples Correlations**
   - Correlation heatmaps show tentative relationships between stock returns and multiples.
   - Beta+ / Beta- correlations with last multiples hint at possible predictive power of valuation metrics for risk in different market conditions.

3. **Tables and Summaries**
   - Key multiples per company are displayed in a concise table.
   - Combined CAPM + multiples table allows a cross-sectional view of risk and valuation.

4. **Speculative Insights**
   - Tech companies may exhibit pro-cyclical risk characteristics, amplifying returns during up-markets.
   - Banks show more muted asymmetry in Beta, possibly reflecting regulatory constraints and asset structure.
   - Investors could potentially leverage the observed Beta-multiple relationships to identify firms whose risk profile aligns with specific market scenarios.

---

## Usage

- Run the Python script to generate:
  - Conditional CAPM scatter plots for each company
  - Heatmaps of correlations between returns, multiples, and Beta
  - Tables summarizing last multiples and combined CAPM + multiples
- The results provide a **diagnostic framework** for exploratory investment analysis and risk assessment.

---

## References

1. Sharpe, W.F. (1964). *Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk*. Journal of Finance.  
2. Fama, E.F., French, K.R. (1993). *Common risk factors in the returns on stocks and bonds*. Journal of Financial Economics.  
3. Yahoo Finance API (`yfinance`) documentation for financial data extraction.
