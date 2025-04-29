# %% [markdown]
#
# # `skrub` - where to we head to?
#
# ## A new way to assemble data
#
# WIP: https://github.com/skrub-data/skrub/pull/1233
#
# If you have a database with several tables, then you will need to assemble them.
#
# We will see that using tools as `pandas` or `polars` when dealing with machine
# learning processes is far to be easy.
#
# Let's look at such a dataset:

# %%
import skrub
from skrub.datasets import fetch_credit_fraud

dataset = fetch_credit_fraud()
skrub.TableReport(dataset.baskets)

# %%
skrub.TableReport(dataset.products)

# %%
# An example of basket looks like this
next(iter(dataset.products.groupby("basket_ID")))[1]

# %% [markdown]
# So let's develop a predictive model to predict whether a basket is fraudulent or not.
#
# We can use the `TableVectorizer` to vectorize the strings in this dataset.

# %%
vectorizer = skrub.TableVectorizer(high_cardinality=skrub.StringEncoder(), n_jobs=-1)
vectorized_products = vectorizer.fit_transform(dataset.products)
vectorized_products

# %% [markdown]
#
# We can now aggregate the products and join the tables: pandas operations.

# %%
aggregated_products = vectorized_products.groupby("basket_ID").agg("mean").reset_index()
aggregated_products

# %%
baskets = dataset.baskets.merge(
    aggregated_products, left_on="ID", right_on="basket_ID"
).drop(columns=["ID", "basket_ID"])
baskets

# %% [markdown]
#
# Great I have a dataset on which I can train a model.

# %%
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_validate

y = baskets["fraud_flag"]
X = baskets.drop("fraud_flag", axis=1)

model = ExtraTreesClassifier(n_jobs=-1)
cv_results = cross_validate(model, X, y, scoring="roc_auc", return_train_score=True)
cv_results = pd.DataFrame(cv_results)

# %%
cv_results[["train_score", "test_score"]].aggregate(["mean", "std"])

# %% [markdown]
#
# But things are ready to go sideways:
# We're in pandas' land. When comes new data, how to apply the same transformations?
# How to cross-validate, or tune the data-preparation steps?
#
# `scikit-learn` fanatics would say: "use pipelines".
#
# But the current `scikit-learn` pipelines do not easily go back up to the data source.
#
# So let's see what `skrub` envisions for this.
#
# We define our inputs as "variables": you can see them as the "source" of the data.
# They are symbolic, meaning that they will allow to record the transformations applied
# to the data. Additionally, we are able to pass a concrete datasets to them such that
# we have an eager evaluation of the transformations to see if what we do works as
# expected.

# %%
products = skrub.var("products", dataset.products)
products

# %% [markdown]
#
# Now we define our "X" and "y" variables.

# %%
baskets = skrub.var("baskets", dataset.baskets)
basket_IDs = baskets[["ID"]].skb.mark_as_X()
fraud_flags = baskets["fraud_flag"].skb.mark_as_y()

# %% [markdown]
#
# `skrub` provides a `polars`-like API to select columns.

# %%
from skrub import selectors as s

vectorized_products = products.skb.apply(vectorizer, cols=s.all() - "basket_ID")
vectorized_products

# %% [markdown]
#
# We aggregate the products

# %%
aggregated_products = vectorized_products.groupby("basket_ID").agg("mean").reset_index()
aggregated_products

# %% [markdown]
#
# And we join the tables.

# %%
features = basket_IDs.merge(aggregated_products, left_on="ID", right_on="basket_ID")
features = features.drop(columns=["ID", "basket_ID"])
features

# %% [markdown]
#
# And we do the prediction

# %%
from sklearn.ensemble import ExtraTreesClassifier

predictions = features.skb.apply(ExtraTreesClassifier(n_jobs=-1), y=fraud_flags)
predictions

# %% [markdown]
# What's the big deal? We now have a graph of computations
# We can apply it to new data
#
# We load the test data

# %%
data_test = fetch_credit_fraud(split="test")
y_test = data_test.baskets["fraud_flag"]

# %%
basket_test = data_test.baskets.drop("fraud_flag", axis=1)

# %% [markdown]
#
# We can apply a predictor to this new data.

# %%
help(predictions.skb.get_estimator)

# %%
predictor = predictions.skb.get_estimator(fitted=True)

# %%
from sklearn.metrics import classification_report

y_pred = predictor.predict(
    {
        "baskets": basket_test,
        "products": data_test.products,
    }
)
print(classification_report(y_test, y_pred))

# %% [markdown]
#
# We can also tune hyperparameters of our data preparation. We just need to
# change a bit the above code.

# %%
encoder = skrub.StringEncoder(
    vectorizer=skrub.choose_from(["tfidf", "hashing"], name="vectorizer"),
)
vectorizer = skrub.TableVectorizer(high_cardinality=encoder, n_jobs=2)
extra_trees = ExtraTreesClassifier(
    max_leaf_nodes=skrub.choose_from([10, 30, 100], name="max_leaf_nodes"),
    n_jobs=-1,
)

# %% [markdown]
#
# The rest of the code remains the same

# %%
vectorized_products = products.skb.apply(vectorizer, cols=s.all() - "basket_ID")
aggregated_products = vectorized_products.groupby("basket_ID").agg("mean").reset_index()

# %% [markdown]
#
# We redefine our sources, to have a clean start

# %%
from pathlib import Path
import joblib

predictions_path = Path("../data/02_predictions.joblib")
search_path = Path("../data/02_search.joblib")

if predictions_path.exists():
    predictions = joblib.load(predictions_path)
    search = joblib.load(search_path)
else:
    baskets = skrub.var("baskets", dataset.baskets)
    basket_IDs = baskets[["ID"]].skb.mark_as_X()
    fraud_flags = baskets["fraud_flag"].skb.mark_as_y()
    features = basket_IDs.merge(aggregated_products, left_on="ID", right_on="basket_ID")
    features = features.drop(columns=["ID", "basket_ID"])
    predictions = features.skb.apply(extra_trees, y=fraud_flags)
    search = predictions.skb.get_grid_search(fitted=True, scoring="roc_auc", verbose=2)

# %%
pd.DataFrame(search.cv_results_)

# %% [markdown]
#
# `skrub` gives you all kinds of tools to tune and inspect this pipeline:
# For instance, we can visualize the hyperparameters selection.

# %%
search.plot_results()

# %% [markdown]
#
# We can also get a full report of the pipeline

# %%
predictions.skb.full_report()
