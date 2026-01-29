#%%
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#%% [markdown]
# ###Job Placement Dataset
# Question: Can we predict if a student will be placed in a job based on their work experience, specialization, and school performance?
# Independent Business Metric: Placement Rate
# Higher placement rates improve rankings of institutions, student satisfaction, and future enrollment.
#%%
job_url = (
    "https://raw.githubusercontent.com/DG1606/CMS-R-2020/"
    "master/Placement_Data_Full_Class.csv"
)
job = pd.read_csv(job_url)

job.head()

# %%
job.info()

# %%
job = job.loc[job.notna().all(axis=1)]

# %%
cat_cols = [
    'gender', 'ssc_b', 'hsc_b', 'hsc_s',
    'degree_t', 'workex', 'specialisation', 'status'
]

job[cat_cols] = job[cat_cols].astype('category')

# %%
job['specialisation'] = job['specialisation'].apply(
    lambda x: x if x in ['Mkt&HR', 'Mkt&Fin'] else 'Other'
).astype('category')

# %%
job_encoded = pd.get_dummies(job, columns=cat_cols)
job_encoded.info()

# %%
num_cols = list(job_encoded.select_dtypes('number'))

scaler = MinMaxScaler()
job_encoded[num_cols] = scaler.fit_transform(job_encoded[num_cols])

# %%
job_encoded['placed_f'] = job_encoded['status_Placed']
job_encoded = job_encoded.drop(
    columns=['status_Placed', 'status_Not Placed', 'sl_no', 'salary']
)

# %%
prevalence_job = job_encoded.placed_f.value_counts(normalize=True)[1]
print(f"Job Placement Prevalence: {prevalence_job:.2%}")

# %%
train_job, temp_job = train_test_split(
    job_encoded,
    train_size=0.7,
    stratify=job_encoded.placed_f,
    random_state=42
)

tune_job, test_job = train_test_split(
    temp_job,
    train_size=0.5,
    stratify=temp_job.placed_f,
    random_state=42
)

print(train_job.shape, tune_job.shape, test_job.shape)

# %% [markdown]
# I am worried about the risk of bias from gender and the academic background. I dropped the salary information as well because of the amount of nulls, which may affect results differently. I also like to work with integers over floats.
# %%
def clean_job_data(df):
    return df.loc[df.notna().all(axis=1)]


def encode_job_data(df):
    cat_cols = [
        'gender', 'ssc_b', 'hsc_b', 'hsc_s',
        'degree_t', 'workex', 'specialisation', 'status'
    ]
    df[cat_cols] = df[cat_cols].astype('category')
    df['specialisation'] = df['specialisation'].apply(
        lambda x: x if x in ['Mkt&HR', 'Mkt&Fin'] else 'Other'
    ).astype('category')
    return pd.get_dummies(df, columns=cat_cols)


def scale_job_data(df):
    num_cols = list(df.select_dtypes('number'))
    df[num_cols] = MinMaxScaler().fit_transform(df[num_cols])
    return df


def create_job_target(df):
    df['placed_f'] = df['status_Placed']
    return df.drop(
        columns=['status_Placed', 'status_Not Placed', 'sl_no', 'salary']
    )


def split_job_data(df):
    train, temp = train_test_split(
        df, train_size=0.7, stratify=df.placed_f, random_state=42
    )
    tune, test = train_test_split(
        temp, train_size=0.5, stratify=temp.placed_f, random_state=42
    )
    return train, tune, test

# %%
def job_pipeline(raw_df):
    df = clean_job_data(raw_df)
    df = encode_job_data(df)
    df = scale_job_data(df)
    df = create_job_target(df)
    return split_job_data(df)

# %% [markdown]
# ### College Completion Dataset
# Question: Can we predift if a college as a high graduation rate based on the characteristics of the institution and student composition?
# Independent Business Metric: Graduation Rate
# This impacts ranking, reputation, and funding.
# %%
college = pd.read_csv("cc_institution_details.csv")

college.head()

college.shape
# %%
college_clean = college.drop(
    columns=['index', 'unitid', 'chronname', 'city', 'site', 'carnegie_ct', 'nicknames'],
    errors='ignore'
)

# %%
cat_cols = ['state', 'level', 'control', 'basic']  
college_clean[cat_cols] = college_clean[cat_cols].astype('category')

# %%
college_encoded = pd.get_dummies(college_clean, columns=cat_cols)
college_encoded.info()

# %%
num_cols = list(college_encoded.select_dtypes('number'))

scaler = MinMaxScaler()
college_encoded[num_cols] = scaler.fit_transform(college_encoded[num_cols])

# %%
college_encoded['grad_f'] = pd.cut(
    college_encoded.grad_150_value,
    bins=[0, 0.5, 1],
    labels=[0, 1]
)

college_encoded['grad_f'] = pd.cut(
    college_encoded.grad_150_value,
    bins=[0, 0.5, 1],
    labels=[0, 1]
)
college_encoded = college_encoded.dropna(subset=['grad_f'])
college_encoded = college_encoded.drop(columns=['grad_150_value'])
# %%
prevalence_college = college_encoded.grad_f.value_counts(normalize=True)[1]
print(f"College Graduation Prevalence: {prevalence_college:.2%}")


# %%
train_col, temp_col = train_test_split(
    college_encoded,
    train_size=0.7,
    stratify=college_encoded.grad_f,
    random_state=42
)

tune_col, test_col = train_test_split(
    temp_col,
    train_size=0.5,
    stratify=temp_col.grad_f,
    random_state=42
)

print(train_col.shape, tune_col.shape, test_col.shape)

# %% [markdown]
# There are a lot of features and lots of dropping. Some variables are akin to socioeconomic differences and public/private differences.

# %%
def clean_college_data(df):
    return df.drop(
        columns=['index', 'unitid', 'chronname', 'city', 'site', 'carnegie_ct', 'nicknames'],
        errors='ignore'
    )

def encode_college_data(df):
    cat_cols = ['state', 'level', 'control', 'basic']
    df[cat_cols] = df[cat_cols].astype('category')
    return pd.get_dummies(df, columns=cat_cols)

def scale_college_data(df):
    num_cols = list(df.select_dtypes('number'))
    df[num_cols] = MinMaxScaler().fit_transform(df[num_cols])
    return df

def create_college_target(df):
    df['grad_f'] = pd.cut(
        df.grad_150_value,
        bins=[0, 0.5, 1],
        labels=[0, 1]
    )
    return df.drop(columns=['grad_150_value'])

def split_college_data(df):
    train, temp = train_test_split(
        df, train_size=0.7, stratify=df.grad_f, random_state=42
    )
    tune, test = train_test_split(
        temp, train_size=0.5, stratify=temp.grad_f, random_state=42
    )
    return train, tune, test

# %%
def college_pipeline(raw_df):
    df = clean_college_data(raw_df)
    df = encode_college_data(df)
    df = scale_college_data(df)
    df = create_college_target(df)
    return split_college_data(df)

# %%
