from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

# Load built-in MovieLens 100k dataset
data = Dataset.load_builtin('ml-100k')

# Build and configure the SVD algorithm
algo = SVD()

# Evaluate using cross-validation with RMSE and MAE
results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Print average RMSE and MAE over the folds
print("Average RMSE:", sum(results['test_rmse']) / len(results['test_rmse']))
print("Average MAE:", sum(results['test_mae']) / len(results['test_mae']))
