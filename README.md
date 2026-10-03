# End-to-End Wine Quality Prediction & Monitoring Pipeline

An advanced, production-grade machine learning system engineered to predict wine quality profiles. It features object-oriented estimators, multi-branch preprocessing architectures, and a built-in statistical data drift monitoring system.

## Core Engineering Highlights
* **Custom Scikit-Learn Transformers:** Authored an object-oriented `ClusterSimilarity` estimator implementing RBF-kernel mappings over dynamic `KMeans` centroids.
* **Leakage-Free Preprocessing:** Isolated feature streams via a composite `ColumnTransformer` to safely execute separate numeric scaling, categorical encoding, and non-linear log transformations without data contamination.
* **Automated Data Drift Detection:** Implemented an integrated statistical monitoring framework using custom Z-score calculations to evaluate distribution changes on incoming production data before passing it to inference.
* **Hyperparameter Optimization:** Conducted multi-dimensional tuning via `RandomizedSearchCV` optimizing downstream estimator structures alongside upstream preprocessing parameters simultaneously.

## Performance Matrix
* **Dummy Baseline Model (Mean Guess):** 0.8738 RMSE
* **Linear Regression Baseline:** 0.7254 RMSE (Std Dev: ±0.0244)
* **Support Vector Regressor (SVR) Baseline:** 0.6883 RMSE (Std Dev: ±0.0271)
* **Optimized Random Forest Pipeline (Winner):** 0.6069 RMSE *(~30.5% total error reduction over the standard mean guess baseline)*


## Statistical Validation
The model's generalization capabilities were verified by constructing a **95% Confidence Interval** around the final Root Mean Squared Error (RMSE) using a Student-t distribution calculation adjusted for degrees of freedom (N-1). This mathematical correction ensures real-world error limits are accurately bounded for unseen production data.
