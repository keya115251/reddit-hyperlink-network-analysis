# =====================================
# BASIC CLEANING
# =====================================
def clean_data(df):
    print("[INFO] Cleaning dataset...")

    # Drop missing critical values
    df = df.dropna(subset=[
        'SOURCE_SUBREDDIT',
        'TARGET_SUBREDDIT',
        'TIMESTAMP',
        'PROPERTIES'
    ])

    # Remove duplicates
    df = df.drop_duplicates()

    print(f"[INFO] Shape after cleaning: {df.shape}")
    return df


# =====================================
# CONVERT TYPES
# =====================================
def convert_types(df):
    print("[INFO] Converting data types...")

    # Convert timestamp
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], errors='coerce')
    df = df.dropna(subset=['TIMESTAMP'])

    # Ensure sentiment is integer
    df['LINK_SENTIMENT'] = df['LINK_SENTIMENT'].astype(int)

    return df


# =====================================
# PARSE FEATURE VECTOR
# =====================================
def parse_properties(prop):
    try:
        values = list(map(float, prop.split(',')))
        if len(values) == 86:
            return np.array(values)
        else:
            return np.zeros(86)
    except:
        return np.zeros(86)


def create_feature_vectors(df):
    print("[INFO] Parsing PROPERTIES into feature vectors...")
    df['FEATURE_VECTOR'] = df['PROPERTIES'].apply(parse_properties)

    # Validate
    lengths = df['FEATURE_VECTOR'].apply(len)
    df = df[lengths == 86]

    print(f"[INFO] Valid feature rows: {df.shape}")
    return df


# =====================================
# TIME FEATURE ENGINEERING
# =====================================
def extract_time_features(df):
    print("[INFO] Extracting time-based features...")

    df['year'] = df['TIMESTAMP'].dt.year
    df['month'] = df['TIMESTAMP'].dt.month
    df['day'] = df['TIMESTAMP'].dt.day
    df['hour'] = df['TIMESTAMP'].dt.hour

    return df


# =====================================
# ENCODE SUBREDDITS
# =====================================
def encode_subreddits(df):
    print("[INFO] Encoding subreddit labels...")

    le_source = LabelEncoder()
    le_target = LabelEncoder()

    df['source_encoded'] = le_source.fit_transform(df['SOURCE_SUBREDDIT'])
    df['target_encoded'] = le_target.fit_transform(df['TARGET_SUBREDDIT'])

    return df


# =====================================
# FINAL FEATURE MATRIX
# =====================================
def create_feature_matrix(df):
    print("[INFO] Creating feature matrix...")

    X_text = np.vstack(df['FEATURE_VECTOR'].values)
    X_extra = df[['source_encoded', 'target_encoded', 'year', 'month', 'day', 'hour']].values

    X = np.hstack([X_text, X_extra])
    y = df['LINK_SENTIMENT'].values

    print(f"[INFO] Feature matrix shape: {X.shape}")
    return X, y


# =====================================
# NORMALIZATION
# =====================================
def scale_features(X):
    print("[INFO] Scaling features...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled


# =====================================
# MAIN PIPELINE
# =====================================
def preprocess_pipeline(filepath):
    df = load_dataset(filepath)
    df = clean_data(df)
    df = convert_types(df)
    df = create_feature_vectors(df)
    df = extract_time_features(df)
    df = encode_subreddits(df)

    X, y = create_feature_matrix(df)
    X_scaled = scale_features(X)

    print("[INFO] Preprocessing complete.")
    return df, X_scaled, y


# =====================================
# RUN
# =====================================
df, X_scaled, y = preprocess_pipeline("soc-redditHyperlinks-body.tsv")