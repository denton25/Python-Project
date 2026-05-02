from scipy.io import loadmat

data = loadmat("emnist-balanced.mat")

print(data.keys())
print(type(data["dataset"]))
print(data["dataset"].dtype)
