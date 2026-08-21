# Olist Brazilian E-Commerce Analysis Project

A data analysis portfolio built on the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle), analyzing **customers, delivery, and revenue** as three independent tracks. Customers are approached through segmentation (unsupervised learning), delivery through funnel/retention analysis, and revenue through regression-based forecasting (supervised learning) — one dataset examined through three different methodologies.

## Project structure

| Part | Approach | Key content | Notebook |
|---|---|---|---|
| Customers | Unsupervised (clustering) | 6 RFM-variant features + KMeans -> 5 segments | `olist_customer_clustering_4_EN.ipynb` |
| Delivery | Funnel/statistical analysis | Order-status funnel (purchase→approval→shipment→delivery) reach rate and days-to-stage | `olist_segment_funnel_retention_EN.ipynb` |
| Revenue | Supervised (regression) | Daily revenue regression forecast + 7-day trend + month-end pacing | `olist_daily_sales_forecasting_EN.ipynb`, `olist_forecast_v2_EN.ipynb`, `olist_regional_forecast_EN.ipynb` |

## Folder structure

```
├── olist_customer_clustering_4_EN.ipynb        # 1) Customer segmentation (clustering)
├── olist_segment_funnel_retention_EN.ipynb      # 2) Delivery funnel · retention · cohort deep dive
├── olist_daily_sales_forecasting_EN.ipynb       # 3) Daily revenue regression model (with feature engineering)
├── olist_forecast_v2_EN.ipynb                   # 4) Forecast redesign (rolling-origin CV + conformal prediction interval)
└── olist_regional_forecast_EN.ipynb             # 5) Regional (SP/MG) revenue forecast
```

## 1. Customer segmentation

Olist's repeat-purchase rate is extremely low (~3% overall), which makes the traditional RFM "Frequency" axis meaningless. It's replaced with 6 rebuilt features — **payment amount, review score, review text length (engagement), delivery delay days, freight cost ratio, and cart item count** — then clustered via log transform + standardization + KMeans (k=5), reaching a silhouette score of 0.253.

| Segment | Characteristics | Revenue share | Repeat rate |
|---|---|---:|---:|
| Delivery-Disappointed churn risk | Most delivery delay, lowest rating, long complaint-style reviews | 10.9% | 2.37% |
| Silent Premium | High payment, high satisfaction, but rarely leaves reviews | 43.2% | 3.14% |
| Engaged Premium | Similar payment, leaves long (60+ char) positive reviews | 24.2% | 4.53% |
| Budget-Regular | Low-price, mostly single purchases, the unremarkable majority | 7.1% | 1.44% |
| Bulk-Buyer | Only group with 2+ cart items | 14.6% | 3.58% |

- k and the feature set were finalized by weighing silhouette score, discriminative power, distance ratios, and business interpretability together (8 -> 6 features, k=4 -> k=5).
- The Delivery-Disappointed segment has lower per-customer payment than normal segments, and the difference in repeat rate by segment was confirmed significant via chi-square test (p<0.000001).

## 2. Delivery funnel analysis

The 4-stage funnel (purchase → approval → shipment → delivery) was analyzed by segment, category, customer region, and seller region.

- **The last-mile stage (shipment → delivery) is the biggest bottleneck**, and the gap between segments widens the most here too.
- The regional gap in delivery time is far larger than the category gap (e.g. SP 8.8 days vs. AM 26.43 days — category differences are only around 2 days).
- The overall SLA (estimated delivery date) miss rate is 8.11%, and the ranking by actual delivery speed doesn't fully match the ranking by SLA violation rate (some regions are slow but rarely violate because of generous promised dates, while others are reasonably fast but violate often because of tight promised dates).

## 3. Revenue forecasting

- **Daily revenue regression**: calendar features + lag/moving averages + holidays/Black Friday + daily unique customer/seller counts, 13 features, RandomForest regression. R²=0.569.
- **Forecast redesign (v2)**: a global model with horizon as a feature + rolling-origin cross-validation + Split Conformal Prediction to calibrate the interval. Produces two deliverables: a next-7-day trend and a projected month-end close (pacing).
- On a true out-of-sample basis, the 7-day aggregate revenue error rate varied by origin date (2.75-17.05%), while month-end pacing was comparatively stable — averaging 5.78% error across 14 origin dates (92.9% interval coverage).
- Black Friday showed a revenue spike of about 6.54x versus normal.
- Daily forecasts broken down by category or region were too noisy to beat even the naive baseline (insufficient sample size), so segmented forecasting was abandoned except for SP and MG, with the rest shown only as actuals trends.

## Tech stack

`Python` · `pandas` · `numpy` · `scikit-learn` (KMeans, RandomForest, PCA) · `scipy` (t-test, chi-square, correlation analysis) · `statsmodels` (VIF) · `BeautifulSoup` (holiday scraping) · `matplotlib` / `seaborn` / `plotly` · `Streamlit`

## Limitations and notes

- Some multicollinearity (VIF>10) was found among features in the revenue regression; this doesn't affect forecast performance itself, but the "ranking" of individual feature importances should be read with caution.
- The estimated potential revenue loss for the Delivery-Disappointed segment is a cumulative figure over the dataset's full observation window; differences in each customer's active period and causality were not verified.
- Regional (beyond SP/MG) and category-level revenue forecasts were excluded from the final analysis due to insufficient sample size for reliable performance.
