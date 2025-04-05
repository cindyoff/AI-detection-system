# dataset prep
test['class'].isna().sum()
test = test.dropna(subset=['class'])
df.rename(columns={'Generation':'article'}, inplace=True)
df.rename(columns={'label':'class'}, inplace=True)

# label encoding : 1 = human, 0 = AI
df['binary_label'] = df['class'].apply(lambda x: 1 if 'human' in x else 0)
test['binary_label'] = test['class'].apply(lambda x: 1 if x == 1 else 0)

# dataset shuffling to avoid under-representation when doing the random sampling
test_shuffled = test.sample(frac=1, random_state=42).reset_index(drop=True)

# 30% of df put into the training dataset 
test_split_index = int(len(test_shuffled) * 0.3)
test_part_train = test_shuffled[:test_split_index] # train
test_part_test = test_shuffled[test_split_index:]  # test (70%)

# merge between the 2 datasets
df_with_test = pd.concat([df, test_part_train], ignore_index=True)

# feature engineering
def count_repeated_words(text):
    """number of words within a sentence"""
    word_counts = Counter(text.split())
    return sum(count > 2 for count in word_counts.values())

def punctuation_ratio(text):
    """ratio of punctuation within a sentence"""
    punctuation_count = sum(1 for char in text if char in string.punctuation)
    return punctuation_count / len(text) if len(text) > 0 else 0

def sentence_length_variance(text):
    """variance of a sentence"""
    sentences = text.split('.')
    lengths = [len(sentence.split()) for sentence in sentences if len(sentence) > 0]
    return np.var(lengths) if lengths else 0

def detect_human_markers(text):
    """look for personal pronouns"""
    human_markers = {"i", "me", "my", "mine", "myself"}
    return sum(1 for word in text.split() if word in human_markers)

def count_informal_words(text):
    """look for informal wordings, any wordings that 
    would not fit in an academic environment"""
    informal_words = {"idk", "btw", "u", "gonna", "lemme", "wanna", "y’all", "gimme", "dunno", "lol"}
    return sum(1 for word in text.split() if word in informal_words)

def detect_emotional_tone(text):
    """look for words transcribing any type of emotions"""
    emotion_words = {"love", "hate", "happy", "sad", "excited", "angry", "surprised", "disappointed"}
    return sum(1 for word in text.split() if word in emotion_words)

def count_punctuation_issues(text):
    """look for mistakes in punctuation"""
    return len(re.findall(r'[!?.,]{2,}', text))

def count_hyphen_splits(text):
    """number of words split in the middle by a hyphen"""
    return len(re.findall(r'\w+-\s', text))

def count_abnormal_spaces(text):
    """number of excess spaces between words in a sentence"""
    return len(re.findall(r'\s{2,}|(?<!\s)\s(?!\s)', text))

def add_features(df):
    df['text_length'] = df['article'].astype(str).apply(lambda x: len(x.split()))
    df['repeated_words'] = df['article'].astype(str).apply(count_repeated_words)
    df['punctuation_ratio'] = df['article'].astype(str).apply(punctuation_ratio)
    df['sentence_length_var'] = df['article'].astype(str).apply(sentence_length_variance)
    df['human_markers'] = df['article'].astype(str).apply(detect_human_markers)
    df['informal_words'] = df['article'].astype(str).apply(count_informal_words)
    df['emotional_tone'] = df['article'].astype(str).apply(detect_emotional_tone)
    df['punctuation_issues'] = df['article'].astype(str).apply(count_punctuation_issues)
    df['hyphen_splits'] = df['article'].astype(str).apply(count_hyphen_splits)
    df['abnormal_spaces'] = df['article'].astype(str).apply(count_abnormal_spaces)
    return df

df_with_test = add_features(df_with_test)
test_part_test = add_features(test_part_test)

# dividing dataset into train and test
X_train = df_with_test.drop(columns=['class', 'binary_label'])
y_train = df_with_test['binary_label']
X_test = test_part_test.drop(columns=['class', 'binary_label'])
y_test = test_part_test['binary_label']

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train['article'].astype(str))
X_test_tfidf = vectorizer.transform(X_test['article'].astype(str))

# merge TF-IDF and the features
X_train_manual = csr_matrix(X_train.drop(columns=['article']).values.astype(np.float64))
X_test_manual = csr_matrix(X_test.drop(columns=['article']).values.astype(np.float64))
X_train_full = hstack([X_train_tfidf, X_train_manual])
X_test_full = hstack([X_test_tfidf, X_test_manual])

# SMOTE (synthetic minority oversampling technique)
if len(y_train.unique()) > 1:
    sm = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = sm.fit_resample(X_train_full, y_train)
else:
    X_train_balanced, y_train_balanced = X_train_full, y_train
