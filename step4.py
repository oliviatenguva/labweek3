# %%
# Import packages
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# %% [markdown]
# ### Job Placement Pipeline
# %%
# Remove rows with missing values
def clean_job_data(df):
    return df.loc[df.notna().all(axis=1)]

# Convert categorical variables
def encode_job_data(df):
    categorical_columns = [
        "gender", "ssc_b", "hsc_b", "hsc_s",
        "degree_t", "workex", "specialisation", "status"
    ]

    df[categorical_columns] = df[categorical_columns].astype("category")
# One-hot encoding
    df["specialisation"] = df["specialisation"].apply(
        lambda x: x if x in ["Mkt&HR", "Mkt&Fin"] else "Other"
    ).astype("category")

    return pd.get_dummies(df, columns=categorical_columns)

# Normalize numeric features with Min-Max scaling
def scale_job_data(df):
    numeric_columns = list(df.select_dtypes("number"))
    df[numeric_columns] = MinMaxScaler().fit_transform(df[numeric_columns])
    return df

# Create target: Placement
def create_job_target(df):
    df["placed_f"] = df["status_Placed"]
    return df.drop(
        columns=["status_Placed", "status_Not Placed", "sl_no", "salary"]
    )

# Train, Tune, Test split
def split_job_data(df):
    train, temp = train_test_split(
        df, train_size=0.7, stratify=df["placed_f"], random_state=42
    )

    tune, test = train_test_split(
        temp, train_size=0.5, stratify=temp["placed_f"], random_state=42
    )

    return train, tune, test

# Pipeline for data prep
def job_pipeline(raw_df):
    df = clean_job_data(raw_df)
    df = encode_job_data(df)
    df = scale_job_data(df)
    df = create_job_target(df)
    return split_job_data(df)


# %% [markdown]
# ### College Completion Pipeline
# Create target: graduation rate
def create_college_target(df):
    threshold = df["grad_150_value"].median()
    df["high_grad_rate"] = (df["grad_150_value"] >= threshold).astype(int)
    return df.drop(columns=["grad_150_value"])

# Handle missing values by filling them in
def handle_college_missing_values(df):
    """Impute missing values."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in numeric_cols:
        if col != "high_grad_rate":
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df

# Simplify institutional category variables
def collapse_college_categories(df):
    if "basic" in df.columns:
        def simplify(x):
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

        df["basic_category"] = df["basic"].apply(simplify)
        df = df.drop(columns=["basic"])
# HBCU one-hot
    if "hbcu" in df.columns:
        df["is_hbcu"] = (df["hbcu"] == "X").astype(int)
        df = df.drop(columns=["hbcu"])
#Flagship one-hot
    if "flagship" in df.columns:
        df["is_flagship"] = (df["flagship"] == "X").astype(int)
        df = df.drop(columns=["flagship"])

    return df

# One-hot encode catergorical variables
def encode_college_data(df):
    return pd.get_dummies(df, drop_first=True)

# Standardize numeric features
def scale_college_data(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    features = [c for c in numeric_cols if c != "high_grad_rate"]

    df[features] = StandardScaler().fit_transform(df[features])
    return df

# Train, Tune, Test split
def split_college_data(df):
    train, temp = train_test_split(
        df, test_size=0.4, stratify=df["high_grad_rate"], random_state=42
    )

    tune, test = train_test_split(
        temp, test_size=0.5, stratify=temp["high_grad_rate"], random_state=42
    )

    return train, tune, test

# Pipeline
def college_pipeline(raw_df):
    df = create_college_target(raw_df)
    df = handle_college_missing_values(df)
    df = collapse_college_categories(df)
    df = encode_college_data(df)
    df = scale_college_data(df)
    return split_college_data(df)
# %%
