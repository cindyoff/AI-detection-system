char_pval = ttest_ind(test[test['class'] == 0]['char_length'], test[test['class'] == 1]['char_length'], equal_var=False).pvalue
word_pval = ttest_ind(test[test['class'] == 0]['word_count'], test[test['class'] == 1]['word_count'], equal_var=False).pvalue

print(f"\nT-test length (characters) — p-value : {char_pval:.4e}")
print("Significant difference" if char_pval < 0.05 else "Non-significant difference")

print(f"\nT-test number of words — p-value : {word_pval:.4e}")
print("Significant difference" if word_pval < 0.05 else "Non-significant difference")
