# new columns
test['char_length'] = test['article'].astype(str).str.len()
test['word_count'] = test['article'].astype(str).str.split().apply(len)
test['sentence_count'] = test['article'].astype(str).str.count(r'[.!?]')

df['char_length'] = df['article'].astype(str).str.len()
df['word_count'] = df['article'].astype(str).str.split().apply(len)
df['sentence_count'] = df['article'].astype(str).str.count(r'[.!?]')

# class proportion
print("Class proportion :")
print(test['class'].value_counts())
print(test['class'].value_counts(normalize=True) * 100)

# descriptive statistics
print("\nDescriptive statistics - length (caracters) :")
print(test.groupby('class')['char_length'].describe())

print("\nStat descriptives - Nombre de mots :")
print(test.groupby('class')['word_count'].describe())

print("\nStat descriptives - Nombre de phrases :")
print(test.groupby('class')['sentence_count'].describe())

# wordclouds
human_texts = " ".join(test[test['class'] == 0]['article'].dropna().astype(str))
ai_texts = " ".join(test[test['class'] == 1]['article'].dropna().astype(str))

human_wc = WordCloud(width=800, height=400, background_color='white').generate(human_texts)
ai_wc = WordCloud(width=800, height=400, background_color='white').generate(ai_texts)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(human_wc, interpolation='bilinear')
plt.axis('off')
plt.title('Wordcloud - Human-generated')

plt.subplot(1, 2, 2)

# boxplots
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
sns.boxplot(x='class', y='char_length', data=test)
plt.title("text length (caracters)")
plt.xlabel("Source")
plt.ylabel("Number of caracters")

plt.subplot(1, 3, 2)
sns.boxplot(x='class', y='word_count', data=test)
plt.title("Number of words per response")
plt.xlabel("Source")
plt.ylabel("Number of words")

plt.subplot(1, 3, 3)
sns.boxplot(x='class', y='sentence_count', data=test)
plt.title("Number of sentences per response")
plt.xlabel("Source")
plt.ylabel("Number of sentences")

plt.tight_layout()
plt.show()
plt.imshow(ai_wc, interpolation='bilinear')
plt.axis('off')
plt.title('Wordcloud - AI-generated')
plt.tight_layout()
plt.show()
