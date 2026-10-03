import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
df = pd.read_csv('/Users/elshadalizada/Downloads/wine-quality-white-and-red.csv')

print(df.describe())
df.info()
print(df.isna().mean()*100)

#checking the distributions
df.hist(bins=50, figsize=(6,12))

#confirming the features where log transformation is the must (right-skewed distribution) if value is more than 1, it needs logtransformation.
print(df[['residual sugar', 'chlorides', 'volatile acidity', 'sulphates']].skew())

#checking if rare extreme values in quality (feature we split stratify split on are not containing 1 value, as it may break our split)
print(df['quality'].value_counts())

#stratified split of our data and checking the quality % in each split set
train, test = train_test_split(df, test_size=0.2, stratify=df['quality'], random_state=42)
strat_train = (train['quality'].value_counts(normalize=True)).sort_index()
strat_test = (test['quality'].value_counts(normalize=True)).sort_index()
df_quality = (df['quality'].value_counts(normalize=True)).sort_index()

comparison_report = pd.DataFrame({'Overall quality %': df_quality,
'Strat_train quality %':strat_train,
'Strat_test quality %': strat_test})
print(comparison_report.round(4))

#target separation steps
X_train, y_train = train.drop('quality', axis=1), train['quality']
X_test, y_test = test.drop('quality', axis=1), test['quality']

#keeping a copy
wine_train = X_train.copy()

#featureengineering
X_train['sugar_to_alcohol_ratio'] = (X_train['residual sugar']/ X_train['alcohol'])
X_train['free_to_total_sulfur_ratio'] = X_train['free sulfur dioxide'] / (X_train['total sulfur dioxide'] + 1e-6)
X_train['acidity_balance_ratio'] = X_train['fixed acidity'] / X_train['volatile acidity']

X_test["sugar_to_alcohol_ratio"] = X_test["residual sugar"] / X_test["alcohol"]
X_test["free_to_total_sulfur_ratio"] = X_test["free sulfur dioxide"] / (X_test["total sulfur dioxide"] + 1e-6)
X_test["acidity_balance_ratio"] = X_test["fixed acidity"] / X_test["volatile acidity"]

class DataDriftMonitor():
    def __init__(self, X_train: pd.DataFrame):
        self.baseline_means = X_train.mean(numeric_only=True)
        self.baseline_stds = X_train.std(numeric_only=True).replace(0,1e-6)
    def check_drift(self, X_production: pd.DataFrame, threshold_z : float = 2.0):
        prod_means = X_production.mean(numeric_only=True)
        drift_report = {}
        drift_detected = False
    
        print('\n' + '='* 60)
        print('RUNNING DATA DRIFT SYSTEM OVERVIEW')
        print('=' * 60)
        for feature in self.baseline_means.index:
            if feature in prod_means:

                z_score_shift = abs((prod_means[feature] - self.baseline_means[feature]) / self.baseline_stds[feature])
                has_drifted = z_score_shift > threshold_z

                drift_report[feature] = {
                    'baseline_mean' : round(self.baseline_means[feature],3),
                    'production_mean' : round(prod_means[feature],3),
                    'shift_z_score' : round(z_score_shift,2),
                    'drift_detected' : has_drifted
                }
                if has_drifted:
                    print(f'FEATURE ALERT: {feature} drifted by {round(z_score_shift,2)} standard deviations!')
                    drift_detected = True
                else:
                    print(f'Stable {feature:<25} | Z-shift: {round(z_score_shift, 2)}')

            print('=' * 60)
            if drift_detected:
                print('🚨 OVERALL STATUS: DATA DRIFT DETECTED! Your model may be decaying.')
            else:
                print('✅ OVERALL STATUS: NORMAL. The live data matches your training history.')
            print('=' * 60 + '\n')

        return drift_detected, pd.DataFrame(drift_report).T

monitor = DataDriftMonitor(X_train)
is_drifted, drift_results_df = monitor.check_drift(X_test)

#correlationmatrix
features_with_target = pd.concat([X_train, y_train], axis=1)
corr_matrix = features_with_target.corr(numeric_only=True)
print(corr_matrix['quality'].sort_values(ascending=False).drop('quality'))


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectFromModel
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVR

skewed_features = ['residual sugar', 'chlorides', 'volatile acidity', 'sulphates','sugar_to_alcohol_ratio','free_to_total_sulfur_ratio','acidity_balance_ratio']
cat_features = X_train.select_dtypes(include=['object', 'category']).columns
numerical_features = [col for col in X_train.columns if col not in cat_features and col not in skewed_features and col != 'total sulfur dioxide']

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state
    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state = self.random_state)
        self.kmeans_.fit(X, sample_weight = sample_weight)
        return self
    def transform(self,X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    def get_feature_names_out(self, names=None):
        return [f'Cluster {i} similarity' for i in range(self.n_clusters)]

cluster_similarity = ClusterSimilarity(n_clusters=10, gamma=1, random_state=42)

skewed_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')),
('log', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
('scaler', StandardScaler())])

num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')),
('scaler', StandardScaler())])

categorical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

transformer = ColumnTransformer([('skew_branch', skewed_pipeline, skewed_features),
('norm_branch', num_pipeline, numerical_features),
('cat_branch', categorical_pipeline, cat_features),
('cluster', cluster_similarity, ['total sulfur dioxide'])])
from sklearn.metrics import root_mean_squared_error

#CROSS VALIDATION SCORE 
#setting up baseline pipelines for cross-validation 
lr_pipeline = Pipeline([('transformer', transformer),
('linear_model', LinearRegression())])

rf_pipeline = Pipeline([('transformer', transformer),
('random_forest', RandomForestRegressor(random_state=42))])

svr_pipeline = Pipeline([
    ('transformer', transformer),
    ('final_scaler', StandardScaler()), 
    ('svr', SVR(kernel='rbf'))
])
from sklearn.model_selection import cross_val_score

lr_scores =  cross_val_score(lr_pipeline, X_train, y_train, scoring='neg_root_mean_squared_error', cv=10, n_jobs=-1)
rf_scores = cross_val_score(rf_pipeline, X_train, y_train, scoring = 'neg_root_mean_squared_error', cv=10, n_jobs=-1)
svr_scores = cross_val_score(svr_pipeline, X_train, y_train, scoring ='neg_root_mean_squared_error', cv=10, n_jobs=-1)

lr_rmse_scores = -lr_scores
rf_rmse_scores = -rf_scores
svr_rmse_scores = -svr_scores

print('Baseline Performance Comparison RMSE ---')
print(f'Linear Regression : Mean RMSE = {lr_rmse_scores.mean():.4f} (+/-{lr_rmse_scores.std():.4f})')
print(f'Random Regression : Mean RMSE = {rf_rmse_scores.mean():.4f} (+/-{rf_rmse_scores.std():.4f})')
print(f'SVR : Mean RMSE = {svr_rmse_scores.mean():.4f} (+/- {svr_rmse_scores.std():.4f})')

print('-'*50)

full_pipeline = Pipeline([('transformer', transformer),
('random_forest', RandomForestRegressor(random_state=42))])

param_grids = {'transformer__cluster__n_clusters': randint(2,20),
'random_forest__max_features': randint(3,15)}

rnd_search = RandomizedSearchCV(full_pipeline, param_grids, scoring='neg_root_mean_squared_error', n_iter=10, cv=15, random_state=42, n_jobs=-1)

rnd_search.fit(X_train, y_train)

print(f'Best Search CV RMSE: {-rnd_search.best_score_:.4f}')
print(f'Best Hyperparameters:', rnd_search.best_params_)

results = pd.DataFrame(rnd_search.cv_results_)
results['rmse'] = -results['mean_test_score']

final_model = rnd_search.best_estimator_
feature_importances = final_model['random_forest'].feature_importances_
predictions = final_model.predict(X_test)
rmse = root_mean_squared_error(y_test, predictions)
print(rmse)

from sklearn.dummy import DummyRegressor
from sklearn.metrics import root_mean_squared_error

dummy = DummyRegressor(strategy='mean')

dummy.fit(X_train, y_train)
dummy_rmse = root_mean_squared_error(y_test, dummy.predict(X_test))

print(f'Dumb baseline RMSE: {dummy_rmse:.4f}')
print(f'Your pipeline RMSE: {-rnd_search.best_score_:.4f}')

import numpy as np
from scipy import stats

squared_errors = (predictions - y_test) ** 2

confidence = 0.95
degrees_of_freedom = len(squared_errors) - 1
mean_squared_error = squared_errors.mean()

confidence_interval = np.sqrt(
    stats.t.interval(
        confidence, 
        df=degrees_of_freedom,
        loc=mean_squared_error,
        scale=stats.sem(squared_errors)
    )
)
print(f"95% Confidence Interval for the RMSE: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]")
print(confidence_interval)



