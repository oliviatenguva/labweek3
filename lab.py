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
# Load in data
job_url = (
    "https://raw.githubusercontent.com/DG1606/CMS-R-2020/"
    "master/Placement_Data_Full_Class.csv"
)
job = pd.read_csv(job_url)

# View data
job.head()

# %%
job.info()

# %%
# Clean the missing values
job = job.loc[job.notna().all(axis=1)]

# %%
# Use the correct variable types
cat_cols = [
    'gender', 'ssc_b', 'hsc_b', 'hsc_s',
    'degree_t', 'workex', 'specialisation', 'status'
]

job[cat_cols] = job[cat_cols].astype('category')

# %%
# Collapse factor levels
job['specialisation'] = job['specialisation'].apply(
    lambda x: x if x in ['Mkt&HR', 'Mkt&Fin'] else 'Other'
).astype('category')

# %%
# One hot encode
job_encoded = pd.get_dummies(job, columns=cat_cols)
job_encoded.info()

# %%
# Normalize the continuous variables
num_cols = list(job_encoded.select_dtypes('number'))

scaler = MinMaxScaler()
job_encoded[num_cols] = scaler.fit_transform(job_encoded[num_cols])

# %%
# Create the target variable
job_encoded['placed_f'] = job_encoded['status_Placed']
job_encoded = job_encoded.drop(
    columns=['status_Placed', 'status_Not Placed', 'sl_no', 'salary']
)

# %%
# Create prevalence
prevalence_job = job_encoded.placed_f.value_counts(normalize=True)[1]
print(f"Job Placement Prevalence: {prevalence_job:.2%}")

# %%
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
# I am worried about the risk of bias from gender and the academic background. I dropped the salary information as well because of the amount of nulls, which may affect results differently. I also like to work with integers over floats.
# %%
# DAG
#Clean the data
def clean_job_data(df):
    return df.loc[df.notna().all(axis=1)]

# Encode the data
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

# Scale the data
def scale_job_data(df):
    num_cols = list(df.select_dtypes('number'))
    df[num_cols] = MinMaxScaler().fit_transform(df[num_cols])
    return df

# Create target
def create_job_target(df):
    df['placed_f'] = df['status_Placed']
    return df.drop(
        columns=['status_Placed', 'status_Not Placed', 'sl_no', 'salary']
    )

# Split/Train, Tune, Test
def split_job_data(df):
    train, temp = train_test_split(
        df, train_size=0.7, stratify=df.placed_f, random_state=42
    )
    tune, test = train_test_split(
        temp, train_size=0.5, stratify=temp.placed_f, random_state=42
    )
    return train, tune, test

# %%
# Pipeline
def job_pipeline(raw_df):
    df = clean_job_data(raw_df)
    df = encode_job_data(df)
    df = scale_job_data(df)
    df = create_job_target(df)
    return split_job_data(df)

# %% [markdown]
# ### College Completion Dataset
# Question: Can we predift if a college has graduation rates above the median based on the characteristics of the institution and student composition?
# Independent Business Metric: Graduation Rate
# This impacts ranking, reputation, and funding.
# %%
college = pd.read_csv("cc_institution_details.csv")

college.head()

college.shape

#%%
# Create target variable for grad success
def create_college_target(college, target_col='grad_150_value', threshold_percentile=50):
    threshold = college[target_col].quantile(threshold_percentile / 100)
    college['high_grad_rate'] = (college[target_col] >= threshold).astype(int)
    college = college.drop(columns=[target_col])
    return college
# %%
# Include relevant columns and drop unnecessary ones
def select_college_feaures(college):
    drop_cols = [
        'index', 'unitid', 'chronname', 'city', 'site', 'nicknames',
        'similar', 'counted_pct', 'long_x', 'lat_y',  # Geographic not useful
        'grad_100_value', 'grad_100_percentile', 'grad_150_percentile',  # Target-related
        'vsa_year',  # All VSA columns have too many missing values
        'vsa_grad_after4_first', 'vsa_grad_elsewhere_after4_first',
        'vsa_enroll_after4_first', 'vsa_enroll_elsewhere_after4_first',
        'vsa_grad_after6_first', 'vsa_grad_elsewhere_after6_first',
        'vsa_enroll_after6_first', 'vsa_enroll_elsewhere_after6_first',
        'vsa_grad_after4_transfer', 'vsa_grad_elsewhere_after4_transfer',
        'vsa_enroll_after4_transfer', 'vsa_enroll_elsewhere_after4_transfer',
        'vsa_grad_after6_transfer', 'vsa_grad_elsewhere_after6_transfer',
        'vsa_enroll_after6_transfer', 'vsa_enroll_elsewhere_after6_transfer'
    ]
    drop_cols = [col for col in drop_cols if col in college.columns]
    college = college.drop(columns=drop_cols)
    return college
# %%
# Deal with missing values
def handle_college_missing_values(college, strategy='median'):
    initial_rows = len(college)
    # Separate numeric and categorical
    numeric_cols = college.select_dtypes(include=[np.number]).columns.tolist()
    if 'high_grad_rate' in numeric_cols:
        numeric_cols.remove('high_grad_rate')
    
    categorical_cols = college.select_dtypes(include=['object']).columns.tolist()
    
    # Handle numeric missing values
    for col in numeric_cols:
        missing_count = college[col].isnull().sum()
        if missing_count > 0:
            if strategy == 'median':
                college[col].fillna(college[col].median(), inplace=True)
            elif strategy == 'mean':
                college[col].fillna(college[col].mean(), inplace=True)
    
    # Handle categorical missing values
    for col in categorical_cols:
        missing_count = college[col].isnull().sum()
        if missing_count > 0:
            college[col].fillna(college[col].mode()[0] if len(college[col].mode()) > 0 else 'Unknown', inplace=True)
    
    final_rows = len(df)
    return college
# %%
# Collapse factor levels
def collapse_college_categories(college):
     if 'basic' in college.columns:
        def simplify_carnegie(x):
            if pd.isna(x):
                return 'Other'
            x = str(x).lower()
            if 'research' in x:
                return 'Research University'
            elif 'doctoral' in x:
                return 'Doctoral/Research'
            elif 'masters' in x:
                return 'Masters University'
            elif 'baccalaureate' in x:
                return 'Baccalaureate College'
            elif 'associates' in x:
                if 'private' in x and 'profit' in x:
                    return 'Associates For-Profit'
                else:
                    return 'Associates Public/Nonprofit'
            elif 'theological' in x or 'bible' in x:
                return 'Religious Institution'
            elif 'art' in x or 'music' in x or 'design' in x:
                return 'Arts School'
            else:
                return 'Other Specialized'
        
        college['basic_category'] = college['basic'].apply(simplify_carnegie)
        college = college.drop(columns=['basic'])
    
    # HBCU: Convert to binary (X vs empty)
        if 'hbcu' in college.columns:
            college['is_hbcu'] = college['hbcu'].apply(lambda x: 1 if x == 'X' else 0)
            college = college.drop(columns=['hbcu'])
    
    # Flagship: Convert to binary
        if 'flagship' in college.columns:
            college['is_flagship'] = college['flagship'].apply(lambda x: 1 if x == 'X' else 0)
            college = college.drop(columns=['flagship'])
    
        return college

# %%
# One hot encode
def one_hot_encode_college(college):
    categorical_cols = college.select_dtypes(include=['object']).columns.tolist()
    
    if len(categorical_cols) == 0:
        return college, []
        
    # Perform one-hot encoding
    college_encoded = pd.get_dummies(college, columns=categorical_cols, drop_first=True)
        
    return college_encoded, categorical_cols

# %%
# Normalize features
def normalize_college_features(college, exclude_cols=['high_grad_rate']):
    numeric_cols = college.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove excluded columns
    cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]
    
    if len(cols_to_scale) == 0:
        return college, None
    
    # Fit and transform
    scaler = StandardScaler()
    college[cols_to_scale] = scaler.fit_transform(college[cols_to_scale])
        
    return college, scaler

# %%
# Split data (Train, Tune, Test)
def split_college_data(college, target_col='high_grad_rate', 
                       train_size=0.6, tune_size=0.2, test_size=0.2, 
                       random_state=42):
    # Separate features and target
    X = college.drop(columns=[target_col])
    y = college[target_col]
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: separate train and tune from temp
    tune_size_adjusted = tune_size / (train_size + tune_size)
    X_train, X_tune, y_train, y_tune = train_test_split(
        X_temp, y_temp, test_size=tune_size_adjusted, 
        random_state=random_state, stratify=y_temp
    )

    
    return {
        'X_train': X_train,
        'X_tune': X_tune,
        'X_test': X_test,
        'y_train': y_train,
        'y_tune': y_tune,
        'y_test': y_test
    }
# %%
# Pipeline
def college_pipeline(college):
    college = create_college_target(college)
    college = select_college_feaures(college)
    college = handle_college_missing_values(college)
    college = collapse_college_categories(college)
    college = one_hot_encode_college(college)
    college = normalize_college_features(college)
    return split_college_data(college)
# %%
