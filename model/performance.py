# prediction and assessment of the test dataset left
y_pred = clf.predict(X_test_full)
print('Classification Report:')
print(classification_report(y_test, y_pred))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predictions')
plt.ylabel('Real values')
plt.title('Confusion matrix')
plt.show()

# variables' feature importance
feature_importance = clf.feature_importances_[-X_train_manual.shape[1]:]
feature_names = X_train.drop(columns=['article']).columns

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_names)
plt.title("Feature importance of each variable")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# ROC curve and AUC value
y_proba = clf.predict_proba(X_test_full)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Courbe ROC (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve - XGBoost model')
plt.legend(loc="lower right")
plt.show()

print(f"Area under the ROC curve (AUC) : {roc_auc:.4f}")
