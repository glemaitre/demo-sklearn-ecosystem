# %% [markdown]
#
# # `Skore` - an abstraction to ease data science project

# %%
import os

os.environ["POLARS_ALLOW_FORKING_THREAD"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# %% [markdown]
#
# Let's open a skore project in which we will be able to store artifacts from our
# experiments.

# %%
import skore

my_project = skore.Project("../data/my_project", if_exists=True)

# %%
from skrub.datasets import fetch_employee_salaries

datasets = fetch_employee_salaries()
df, y = datasets.X, datasets.y

# %%
from skrub import TableReport

table_report = TableReport(datasets.employee_salaries)
table_report

# %% [markdown]
#
# Let's model our problem.

# %%
from skrub import TableVectorizer, TextEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    TableVectorizer(high_cardinality=TextEncoder()),
    RandomForestRegressor(n_estimators=20, max_leaf_nodes=40),
)
model

# %% [markdown]
#
# `skore` provides a couple of tools to ease the evaluation of model:

# %%
from skore import CrossValidationReport

report = CrossValidationReport(estimator=model, X=df, y=y, cv_splitter=5, n_jobs=4)

# %%
report.help()

# %%
import time

start = time.time()
score = report.metrics.r2()
end = time.time()
print(f"Time taken: {end - start:.2f} seconds")

# %%
score

# %%
start = time.time()
score = report.metrics.r2()
end = time.time()
print(f"Time taken: {end - start:.2f} seconds")

# %%
score

# %%
import time

start = time.time()
score = report.metrics.rmse()
end = time.time()
print(f"Time taken: {end - start:.2f} seconds")

# %%
score

# %%
report.cache_predictions(n_jobs=4)

# %%
my_project.put("Random Forest model report", report)

# %%
report = my_project.get("Random Forest model report")
report.help()

# %%
report.metrics.report_metrics(aggregate=["mean", "std"], indicator_favorability=True)

# %%
display = report.metrics.prediction_error()
display.plot(kind="actual_vs_predicted")

# %%
report.estimator_reports_

# %%
estimator_report = report.estimator_reports_[0]
estimator_report.help()

# %%
estimator_report.metrics.prediction_error().plot(kind="actual_vs_predicted")

# %%
list_df = []

for estimator_report, i in zip(report.estimator_reports_, range(len(report.estimator_reports_))):
    feat_perm = estimator_report.feature_importance.feature_permutation(max_samples=50).droplevel(level = 0).T
    feat_perm["model"] = i
    list_df.append(feat_perm)

# %%
import pandas as pd
df_concat = pd.concat(list_df)
# %%
import plotly.express as px
df = px.data.tips()
fig = px.box(df_concat, color = "model", orientation = "h")
fig.show()

# %% [markdown]
#
# ## Conclusions
#
# **Vision**
# - Develop tooling to create data science artifacts
# - Help at following good practices for the problem at hand
# - Help at the collaboration to carry on data science project
#
# **Wrap-up**
# - Provide tools to evaluate predictive models
# - Make some internal magic to reduce user friction
# - Allow for persistence of artifacts
#
# **Roadmap**
# - Cover multiple aspects of the data science life cycles: data, model, etc.
# - Help at creating artifacts dedicated to the problem at hand and the model
# - Reduce the complexity related to code
