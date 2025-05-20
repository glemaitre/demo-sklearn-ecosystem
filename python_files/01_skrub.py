# %% [markdown]
#
# # `skrub`
#
# ## Help in the exploration of the data

# %%
from skrub.datasets import fetch_employee_salaries

employee_salaries = fetch_employee_salaries()
X, y = employee_salaries.X, employee_salaries.y
X.head()

# %%
y.head()

# %%
from skrub import TableReport

table = TableReport(employee_salaries.employee_salaries)
table

# %% [markdown]
#
# ## Help at preprocessing data
#
# `skrub` comes with a set of additional encoders.

# %%
from skrub import DatetimeEncoder, ToDatetime
from sklearn.pipeline import make_pipeline

encoder = make_pipeline(ToDatetime(), DatetimeEncoder())
encoder.fit_transform(X["date_first_hired"])

# %%
from skrub import MinHashEncoder

encoder = MinHashEncoder()
encoder.fit_transform(X["employee_position_title"])

# %% [markdown]
#
# `TableVectorizer` helps at reducing the boilerplate of `ColumnTransformer`.

# %%
from skrub import TableVectorizer

vectorizer = TableVectorizer()
vectorizer

# %% [markdown]
#
# ## Help at getting a good baseline model
#
# `tabular_learner` to help at getting meaningful baselines quickly.

# %%
from skrub import tabular_learner
from sklearn.linear_model import RidgeCV

model = tabular_learner(RidgeCV())
model

# %%
model = tabular_learner("regressor")
model

# %%
import pandas as pd
from sklearn.model_selection import cross_validate

cv_results = cross_validate(model, X, y)
cv_results = pd.DataFrame(cv_results)
cv_results

# %% [markdown]
#
# ## Machine learning going back to the source

# %%
from skrub.datasets import fetch_credit_fraud

dataset = fetch_credit_fraud()
TableReport(dataset.baskets)

# %%
TableReport(dataset.products)

# %% [markdown]
#
# Express data transformations for machine learning pipelines.

# %%
import skrub

products = skrub.var("products", dataset.products)
baskets = skrub.var("baskets", dataset.baskets)
basket_IDs = baskets[["ID"]].skb.mark_as_X()
fraud_flags = baskets["fraud_flag"].skb.mark_as_y()

# %%
from skrub import selectors as s
from sklearn.ensemble import ExtraTreesClassifier

vectorizer = skrub.TableVectorizer(high_cardinality=skrub.StringEncoder(), n_jobs=-1)
predictor = ExtraTreesClassifier(n_jobs=-1)
predictions = (
    basket_IDs.merge(
        products.skb.apply(vectorizer, cols=s.all() - "basket_ID")
        .groupby("basket_ID")
        .agg("mean")
        .reset_index(),
        left_on="ID",
        right_on="basket_ID",
    )
    .drop(columns=["ID", "basket_ID"])
    .skb.apply(predictor, y=fraud_flags)
)
predictions

# %% [markdown]
#
# Revisit the way to define hyperparameters tuning.

# %%
encoder = skrub.StringEncoder(
    vectorizer=skrub.choose_from(["tfidf", "hashing"], name="vectorizer"),
)
vectorizer = skrub.TableVectorizer(high_cardinality=encoder, n_jobs=-1)
predictor = ExtraTreesClassifier(
    max_leaf_nodes=skrub.choose_from([10, 30, 100], name="max_leaf_nodes"),
    n_jobs=-1,
)

# %%
from pathlib import Path
import joblib

search_path = Path("../data/01_search.joblib")

if search_path.exists():
    search = joblib.load(search_path)
else:
    search = (
        basket_IDs.merge(
            products.skb.apply(vectorizer, cols=s.all() - "basket_ID")
            .groupby("basket_ID")
            .agg("mean")
            .reset_index(),
            left_on="ID",
            right_on="basket_ID",
        )
        .drop(columns=["ID", "basket_ID"])
        .skb.apply(predictor, y=fraud_flags)
    ).skb.get_randomized_search(fitted=True, scoring="roc_auc", verbose=2)
    joblib.dump(search, search_path)

# %%
pd.DataFrame(search.cv_results_)

# %%
search.plot_results()

# %% [markdown]
#
# ## Conclusions
#
# **Vision**
# - Less wrangling, more machine learning
# - Bring the world of database closer to machine learning
#
# **Wrap-up**
# - Additional components to assemble, encode, and vectorize data
# - Reduce boilerplate code to get good baseline
# - Broader the scope of scikit-learn pipeline to the database world
#
# **Bold vision**
# - scikit-learn should be the machine learning toolbox with its numerical
#   optimization roots and expertise
# - skrub could be where the data preparation happen with integration with
#   dataframe-like libraries
