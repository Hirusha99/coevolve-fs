import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from coevolve_fs import CoevolutionarySelector

# 1. Prepare your data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Initialize the Selector
# You can pass any scikit-learn classifier here
classifier = GaussianNB()
selector = CoevolutionarySelector(
    classifier=classifier,
    population_size=50, 
    generations=20,
    crossover_prob=0.8,
    mutation_prob=0.02
)

# 3. Run the Co-evolutionary Search
# 'columns_per_subgroup' defines how the feature space is divided
best_mask = selector.fit(
    X_train, y_train, 
    X_test, y_test, 
    columns_per_subgroup=10
)

# 4. Apply the results
selected_features = X_train.columns[best_mask == 1]
print(f"Original feature count: {X_train.shape[1]}")
print(f"Reduced feature count: {len(selected_features)}")
print(f"Selected Features: {list(selected_features)}")

# 5. Final Model Evaluation
final_model = GaussianNB()
final_model.fit(X_train[selected_features], y_train)
score = final_model.score(X_test[selected_features], y_test)
print(f"Final Accuracy with Selected Features: {score:.4f}")
