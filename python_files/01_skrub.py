# %% [markdown]
#
# # `skrub` - less wrangling, more machine learning
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

# %%
from skrub import patch_display

patch_display()  # you can use skrub.unpatch_display() to disable the display

# %%
X

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
from skrub import TextEncoder

encoder = TextEncoder()
encoder.fit_transform(X["employee_position_title"])

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

# %%
cv_results

# %% [markdown]
#
# Table joiner compatible with the scikit-learn API.

# %%
from skrub import Joiner

airports = pd.DataFrame(
    {
        "airport_id": [1, 2],
        "airport_name": ["Charles de Gaulle", "Aeroporto Leonardo da Vinci"],
        "city": ["Paris", "Roma"],
    }
)
# notice the "Rome" instead of "Roma"
capitals = pd.DataFrame(
    {"capital": ["Berlin", "Paris", "Rome"], "country": ["Germany", "France", "Italy"]}
)
joiner = Joiner(
    capitals,
    main_key="city",
    aux_key="capital",
    max_dist=0.8,
    add_match_info=False,
)
joiner.fit_transform(airports)
