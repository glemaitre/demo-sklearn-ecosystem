# %% [markdown]
#
# # `skops`
#
# Disclaimer: `skops` is not a model registry as for instance MLflow. The vision is
# to provide building blocks that ultimately are useful to move `scikit-learn` models
# closer to production. `skops` would be a so-called "flavor" on which MLflow relies
# on: https://github.com/mlflow/mlflow/issues/13077
#
# ## Storing and loading a model using pickling

# %%
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)

# %%
import joblib

joblib.dump(model, "../data/03_model.joblib")

# %%
model = joblib.load("../data/03_model.joblib")

# %%
print(f"Accuracy: {model.score(X_test, y_test):.2f}")

# %% [markdown]
#
# So what's wrong with this approach?
#
# In short, nothing if you are a single-user environment because:
#
# - you trust the source of the pickle file
# - you handle yourself the environment where the pickle file is loaded

# %%
import pickle
import pickletools


class D:
    def __reduce__(self):
        return (print, ("!!!I SEE YOU!!!",))


pickled = pickle.dumps(D())
pickletools.dis(pickled)

# %%
pickle.loads(pickled)

# %%
import os


class E:
    def __reduce__(self):
        return (
            os.system,
            ("""mkdir -p /tmp/dumps && echo "!!!I'm in YOUR SYSTEM!!!" > /tmp/dumps/demo.txt""",),
        )


pickled = pickle.dumps(E())
pickletools.dis(pickled)

# %%
pickle.loads(pickled)

# %%
with open("/tmp/dumps/demo.txt", "r") as f:
    print(f.read())

# %% [markdown]
#
# ## Securing the pickling process using `skops`
#
# The idea would be to traverse the pickled object and ensure that there is no
# structure or object that we don't explicitly trust.
#
# So by default, `skops` trust scikit-learn objects and we can do the same as with
# `joblib`.

# %%
from skops import io as sio

sio.dump(model, "../data/03_model.skops")

# %%
model_loaded = sio.load("../data/03_model.skops")

# %%
print(f"Accuracy: {model_loaded.score(X_test, y_test):.2f}")

# %% [markdown]
#
# Before to load the model, we can visualize what is trusted or not by `skops`.

# %%
sio.visualize("../data/03_model.skops")

# %% [markdown]
#
# Now, let's have an example where we have a custom object that we don't trust by
# default.

# %%
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline


class TransformerNoOp(BaseEstimator):
    def fit(self, X, y):
        self.fitted_ = True
        return self
    def transform(self, X):
        return X


model = make_pipeline(TransformerNoOp(), DecisionTreeClassifier(random_state=42))
model.fit(X_train, y_train)
sio.dump(model, "../data/03_model.skops")

# %%
sio.visualize("../data/03_model.skops")

# %% [markdown]
#
# We see that one of the object is treated as unsafe. Let's try to load the model.

# %%
sio.load("../data/03_model.skops")

# %% [markdown]
#
# But we are sure of what we handle, then we can load the model with the `trusted`
# parameter.

# %%
sio.load("../data/03_model.skops", trusted=["__main__.TransformerNoOp"])

# %% [markdown]
#
# ## Documenting machine learning models
#
# `skops` allows to create model cards that should be seen as the documentation of
# the model. A model card would be a bunch of written information to understand how
# the model as been trained, for which purpose, and quantitative and qualitative
# analysis of the model. Perfectly, this information should allow to reproduce the
# model.

# %%
from skops.card import Card

model_card = Card(model)

# %%
print(model_card.render())

# %%
limitations = "This model is not ready to be used in production."
model_description = "It is a dummy decision tree just to show a demo."
model_card_authors = "me"

model_card = model_card.add(
    **{
        "Model Card Authors": model_card_authors,
        "Model description": model_description,
        "Model description/Intended uses & limitations": limitations,
    }
)

# %%
print(model_card.render())

# %% [markdown]
#
# To provide model cards in use on open-hubs, we can have a look at the model card
# on Hugging Face: https://huggingface.co/apprentissage-sirius/verbatims-gem
#
# ## Conclusions
#
# **Vision**
# - Develop building blocks to bring scikit-learn closer to production
#
# **Wrap-up**
# - Secure the persistence of scikit-learn models
# - Document scikit-learn models in production via model cards
