from torch.utils.data import random_split

# Define the dataset
dataset = ClipCocoDataset("/kaggle/input/karapathyv5/oscar_split_ViT-B_32_train.pkl", prefix_length=10)

# Calculate split sizes
train_size = int(0.85* len(dataset))
test_size = len(dataset) - train_size

# Split the dataset
train_data, test_data = random_split(dataset, [train_size, test_size])

# Print lengths
print(f"Train size: {len(train_data)}")
print(f"Test size: {len(test_data)}")


from torch.utils.data import random_split
new_dataset = test_data
# Define the dataset
# dataset = ClipCocoDataset("/kaggle/input/karapathyv5/oscar_split_ViT-B_32_train.pkl", prefix_length=4)

# Calculate split sizes
train_size = int(0.95* len(new_dataset))
test_size = len(new_dataset) - train_size

# Split the dataset
train_data, test_data = random_split(new_dataset, [train_size, test_size])

# Print lengths
print(f"Train size: {len(train_data)}")
print(f"Test size: {len(test_data)}")

