# retail-sales-forecasting

End-to-end, production-style time-series forecasting app for retail/SaaS demand planning.
Upload daily sales with date, store, item, sales or use the built-in synthetic generator.
The app trains a pipeline (XGBoost/GB + feature engineering), backtests on a hold-out window, and forecasts the next N days. Includes auto-generated plain-English insights for non-technical stakeholders.

Live demo (Hugging Face Space): [https://huggingface.co/spaces/<your-username>/retail-sales-forecasting](https://huggingface.co/spaces/aakash-malhan/retail-sales-forecasting)
<img width="1494" height="854" alt="Screenshot 2025-10-14 162756" src="https://github.com/user-attachments/assets/220b6179-fffe-4337-a170-8941eded9a0c" />
<img width="1432" height="884" alt="Screenshot 2025-10-14 162814" src="https://github.com/user-attachments/assets/88395eb9-f6c6-4c0a-ae12-dd33f15be113" />
<img width="1486" height="337" alt="Screenshot 2025-10-14 162821" src="https://github.com/user-attachments/assets/9689a318-b29c-45aa-bf1a-37059b8306d2" />


Tech stack: Python · Gradio · scikit-learn · XGBoost (CPU) · matplotlib

Why this project matters (Business Impact)

Accurate short-term forecasts drive:

Inventory: reduce stockouts & overstock → lower working capital.
Ops & Staffing: plan labor for peaks (weekends, promos).
Marketing: spot growth/slowdown early; align promo calendars.
Finance: better revenue outlook → realistic targets.
In experiments on the synthetic—but realistic—dataset, the model’s CV/Test RMSE & MAE are reported in-app. For sparse/short series, a robust baseline kicks in (recent-mean forecast) to keep outputs reliable rather than failing noisily.

Extending the Project

Add Holidays:
Join with a holiday table by date and add flags/lead-lags.

Add Price/Promo:
Extend schema (price, promo_flag) and include as features.

Probabilistic Forecasts:
Replace final regressor with LightGBM Quantile Regressor or use pinball loss.

MLOps:

Log runs & metrics (e.g., MLflow).
Register models, set up batch/online prediction.
CI workflow to test app.py and lint.
