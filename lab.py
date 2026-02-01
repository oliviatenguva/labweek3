#%%
# Import packages
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#%% [markdown]
# ### Dataset 1: Job Placement Dataset
# ### Step 1
# Question: Can we predict if a student will be placed in a job based on their work experience, specialization, and academic background?
# Independent Business Metric: Placement Rate
# Higher placement rates improve rankings of institutions, student satisfaction, and future enrollment.
#%%
# Load in dataset
job_url = (
    "https://raw.githubusercontent.com/DG1606/CMS-R-2020/"
    "master/Placement_Data_Full_Class.csv"
)
job = pd.read_csv(job_url)

# View data
job.head()

# %%
job.info()

# %% [markdown]
# ### Step 2

# %%
# Remove rows with missing values
job = job.loc[job.notna().all(axis=1)]

# %%
# Correct variable types
cat_cols = [
    'gender', 'ssc_b', 'hsc_b', 'hsc_s',
    'degree_t', 'workex', 'specialisation', 'status'
]

job[cat_cols] = job[cat_cols].astype('category')

# %%
# Collapse factor levels and keep major specializations
job['specialisation'] = job['specialisation'].apply(
    lambda x: x if x in ['Mkt&HR', 'Mkt&Fin'] else 'Other'
).astype('category')

# %%
# One hot encoding
job_encoded = pd.get_dummies(job, columns=cat_cols)
job_encoded.info()

# %%
# Normalize the numeric variables
num_cols = list(job_encoded.select_dtypes('number'))
scaler = MinMaxScaler()
job_encoded[num_cols] = scaler.fit_transform(job_encoded[num_cols])

# %%
# Create the target variable
# Target if the student was placed
job_encoded['placed_f'] = job_encoded['status_Placed']
# Drop variables that should not be used as features
drop_columns = [
    "status_Placed",
    "status_Not Placed",
    "sl_no",
    "salary"
]

drop_columns = [col for col in drop_columns if col in job_encoded.columns]

job_encoded = job_encoded.drop(columns=drop_columns)

# %%
# Calculate prevalence
prevalence_job = job_encoded["placed_f"].mean()
print(f"Job Placement Prevalence: {prevalence_job:.2%}")

# %%
# Train/Tune/Test split
# Train
train_job, temp_job = train_test_split(
    job_encoded,
    train_size=0.7,
    stratify=job_encoded.placed_f,
    random_state=42
)

# Tune
tune_job, test_job = train_test_split(
    temp_job,
    train_size=0.5,
    stratify=temp_job.placed_f,
    random_state=42
)

print(train_job.shape, tune_job.shape, test_job.shape)

# %% [markdown]
# ### Step 3
# I am worried about the risk of bias from gender and the academic background. I dropped the salary information as well because of the amount of nulls, which may affect results differently. I also like to work with integers over floats. The data is small which could lead to overfitting. Placement decisions can also depend on factors that is not seen in the data.


# %% [markdown]
# ### Dataset 2: College Completion Dataset
# Question: Can we predift if a college has graduation rates above the median based on the characteristics of the institution and student composition?
# Independent Business Metric: Graduation Rate
# Graduation rate impacts ranking, reputation, and funding.
# %%
college = pd.read_csv("cc_institution_details.csv")

college.head()

college.shape

# %%
# ### Step 2
# Create the target variable
# Target: high graduation rate = above median 150% graduation rate
threshold = college["grad_150_value"].median()
college["high_grad_rate"] = (college["grad_150_value"] >= threshold).astype(int)
college = college.drop(columns=["grad_150_value"])

# %%
# Drop columns that are irrelevant
drop_columns = [
    "index", "unitid", "chronname", "city", "site", "nicknames",
    "similar", "counted_pct", "long_x", "lat_y",
    "grad_100_value", "grad_100_percentile", "grad_150_percentile",
    "vsa_year",
    "vsa_grad_after4_first", "vsa_grad_elsewhere_after4_first",
    "vsa_enroll_after4_first", "vsa_enroll_elsewhere_after4_first",
    "vsa_grad_after6_first", "vsa_grad_elsewhere_after6_first",
    "vsa_enroll_after6_first", "vsa_enroll_elsewhere_after6_first",
    "vsa_grad_after4_transfer", "vsa_grad_elsewhere_after4_transfer",
    "vsa_enroll_after4_transfer", "vsa_enroll_elsewhere_after4_transfer",
    "vsa_grad_after6_transfer", "vsa_grad_elsewhere_after6_transfer",
    "vsa_enroll_after6_transfer", "vsa_enroll_elsewhere_after6_transfer"
]

college = college.drop(
    columns=[c for c in drop_columns if c in college.columns]
)

# %%
# Handle missing values
numeric_cols = college.select_dtypes(include=[np.number]).columns
categorical_cols = college.select_dtypes(include=["object"]).columns

for col in numeric_cols:
    if col != "high_grad_rate":
        college[col] = college[col].fillna(college[col].median())

for col in categorical_cols:
    college[col] = college[col].fillna(college[col].mode()[0])
# %%
# Collapse categories
if "basic" in college.columns:
    def simplify_carnegie(x):
        x = str(x).lower()
        if "research" in x:
            return "Research"
        if "masters" in x:
            return "Masters"
        if "baccalaureate" in x:
            return "Baccalaureate"
        if "associate" in x:
            return "Associate"
        return "Other"

    college["basic_category"] = college["basic"].apply(simplify_carnegie)
    college = college.drop(columns=["basic"])
# HBCU one-hot
if "hbcu" in college.columns:
    college["is_hbcu"] = (college["hbcu"] == "X").astype(int)
    college = college.drop(columns=["hbcu"])
#Flagship one-hot
if "flagship" in college.columns:
    college["is_flagship"] = (college["flagship"] == "X").astype(int)
    college = college.drop(columns=["flagship"])

# %%
# One hot encoding
college = pd.get_dummies(college, drop_first=True)

# %%
# Normalize numeric features
numeric_cols = college.select_dtypes(include=[np.number]).columns
features_to_scale = [c for c in numeric_cols if c != "high_grad_rate"]

college[features_to_scale] = StandardScaler().fit_transform(
    college[features_to_scale]
)

# %%
# Calculate prevalence
prevalence_college = college["high_grad_rate"].mean()
print(f"College Graduation Prevalence (Baseline): {prevalence_college:.2%}")

# %%
# Train, Tune, Test split
train_college, temp_college = train_test_split(
    college,
    test_size=0.4,
    stratify=college["high_grad_rate"],
    random_state=42
)

tune_college, test_college = train_test_split(
    temp_college,
    test_size=0.5,
    stratify=temp_college["high_grad_rate"],
    random_state=42
)

print("College dataset shapes:")
print("Train:", train_college.shape)
print("Tune:", tune_college.shape)
print("Test:", test_college.shape)

# %% [markdown]
# ### Step 3
# There may be bias in institutions like public vs privte. Some of the features are also related to socioeconomic status which can influence results. Graduation rates can be influenced by other factors not shown (like pandemic, etc.). The median split can oversimplify the institutional performance.
# %%
