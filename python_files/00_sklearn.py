# %% [markdown]
#
# # What scikit-learn allows you to do?
#
# No better way than to show an example.

# %%
from sklearn.datasets import fetch_california_housing

data, target = fetch_california_housing(return_X_y=True, as_frame=True)
data.head()

# %%
target.head()

# %%
from sklearn.model_selection import train_test_split

data, data_test, target, target_test = train_test_split(
    data, target, test_size=0.2, random_state=0
)

# %% [markdown]
#
# Craft a predictive model.

# %%
from sklearn.cluster import KMeans
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

geo_columns = ["Latitude", "Longitude"]
spline_columns = ["MedInc", "AveRooms", "AveBedrms", "Population", "AveOccup"]

preprocessor = make_column_transformer(
    (KMeans(n_clusters=10), geo_columns),
    (make_pipeline(StandardScaler(), SplineTransformer()), spline_columns),
)
model = make_pipeline(
    preprocessor,
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=True),
    Ridge(),
)
model

# %% [markdown]
#
# Evaluate the model via cross-validation.

# %%
import pandas as pd
from sklearn.model_selection import cross_validate, ShuffleSplit

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
cv_results = cross_validate(
    model, data, target, cv=cv, return_estimator=True, return_train_score=True
)
cv_results = pd.DataFrame(cv_results)
cv_results[["train_score", "test_score"]].aggregate(["mean", "std"])

# %%
from itertools import zip_longest
import matplotlib.pyplot as plt
from sklearn.metrics import PredictionErrorDisplay

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
for est, ax in zip_longest(cv_results["estimator"], axs.ravel()):
    if est is None:
        ax.set_visible(False)
        continue
    PredictionErrorDisplay.from_estimator(
        est, data_test, target_test, kind="actual_vs_predicted", ax=ax
    )
    ax.set_title(f"R2 score: {est.score(data_test, target_test):.2f}")
plt.tight_layout()

# %% [markdown]
#
# Tune the hyperparameters of the model.

# %%
import numpy as np
from scipy.stats import randint
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RandomizedSearchCV

preprocessor = make_column_transformer(
    (KMeans(n_clusters=10), geo_columns),
    (make_pipeline(StandardScaler(), SplineTransformer()), spline_columns),
)
model = make_pipeline(
    preprocessor,
    PolynomialFeatures(degree=1, include_bias=False, interaction_only=True),
    SelectKBest(k=30),
    RidgeCV(alphas=np.logspace(-5, 5, num=50)),
)

param_distributions = {
    "columntransformer__kmeans__n_clusters": randint(2, 30),
    "columntransformer__pipeline__splinetransformer__n_knots": randint(2, 10),
    "polynomialfeatures__degree": [1, 2],
    "selectkbest__k": randint(50, 1000),
}
search = RandomizedSearchCV(
    model, param_distributions=param_distributions, cv=cv, n_iter=10, verbose=1_000
)

# %%
import joblib
import warnings
from pathlib import Path

cv_results_path = Path("../data/00_search_cv.joblib")

# It is costly, let's reload from the disk if it exists
if cv_results_path.exists():
    cv_results = joblib.load(cv_results_path)
else:
    with warnings.catch_warnings(action="ignore"):
        cv_results = cross_validate(
            search, data, target, cv=cv, return_estimator=True, return_train_score=True
        )
    cv_results = pd.DataFrame(cv_results)
    joblib.dump(cv_results, cv_results_path)
cv_results[["train_score", "test_score"]].aggregate(["mean", "std"])

# %%
for est in cv_results["estimator"]:
    print(est.best_params_)

# %% [markdown]
#
# Bonus point: you can dump the model and use it in production.

# %%
# search.fit(data, target)
# joblib.dump(search.best_estimator_, "../models/00_my_production_model.joblib")
# prod_model = joblib.load("../models/00_my_production_model.joblib")
# prod_model.predict(data)

# %% [markdown]
#
# ## Conclusions
#
# ### Strengths
#
# - Simple consistent API
# - A lot of building block to build and tune your predictive model
# - A lot of tools to evaluate your predictive model
# - A lot of tools to inspect your predictive model
# - Robust and fast implementation
# - Good documentation
#
# ### Pitfalls
#
# **From the demo**
# - By nature, scikit-learn offers generic components
# - Know-how is extremely important
#   - No available baseline to start with
#   - Some syntax are convoluted
#   - Some choices to be made require expertise
#   - One can make methodological errors
#
# **What we did not show**
# - Data preprocessing is actually hard
#   - Data can come from different sources
#   - Transformations are not necessarily standardized
# - What happens once predictive models are in production
#   - Pickling and security
#   - Documentation
#   - Registry
