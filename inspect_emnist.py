from scipy.io import loadmat

data = loadmat("emnist-balanced.mat")

dataset = data["dataset"]

print("\n=== DATASET STRUCTURE ===")
print(dataset.dtype)

train = dataset["train"][0,0]
test = dataset["test"][0,0]
mapping = dataset["mapping"][0,0]

print("\n=== TRAIN STRUCT FIELDS ===")
print(train.dtype)

print("\n=== TEST STRUCT FIELDS ===")
print(test.dtype)

print("\n=== MAPPING SHAPE ===")
print(mapping.shape)

print("\n=== FIRST 10 RAW LABELS ===")
print(train["labels"][0,0].flatten()[:10])

print("\n=== FIRST 10 MAPPING ROWS ===")
print(mapping[:10])
