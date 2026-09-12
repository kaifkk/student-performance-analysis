rf_model = RandomForestModel()
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

evaluator = ModelEvaluator(y_test, y_pred)
evaluator.plot_confusion_matrix()
report = evaluator.evaluate()
report.to_csv('results/random_forest_metrics.csv')

analyzer = ErrorAnalyzer(X_test, y_test, y_pred)
analyzer.save_errors()

interpreter = FeatureInterpreter(rf_model.model, X_train.columns)
interpreter.plot_importance()