# Space-X-Falcon-9-First-Stage-Landing-Prediction-Machine-Learning
# Comparison of all models
models = {
    'Logistic Regression': [logreg_cv.best_score_, logreg_score],
    'SVM': [svm_cv.best_score_, svm_score],
    'Decision Tree': [tree_cv.best_score_, tree_score],
    'KNN': [knn_cv.best_score_, knn_score]
}

# Summary table
summary_df = pd.DataFrame.from_dict(models, orient='index', columns=['Train Accuracy', 'Test Accuracy'])
print(summary_df)

# Visual comparison
summary_df['Test Accuracy'].plot(kind='bar', figsize=(10, 6))
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison')
plt.show()
