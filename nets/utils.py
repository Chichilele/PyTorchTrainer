import matplotlib.pyplot as plt
from torch.utils.data import Dataset


def plot_loss(results):
    train_losses, train_counter = results["train_loss"]
    test_losses, test_counter = results["test_loss"]

    fig = plt.figure()
    plt.plot(train_counter, train_losses, color="blue")
    plt.scatter(test_counter, test_losses, color="red")
    plt.legend(["Train Loss", "Test Loss"], loc="upper right")
    plt.xlabel("number of training examples seen")
    plt.ylabel("negative log likelihood loss")
    return fig


class CustomSubset(Dataset):
    r"""
    Subset of a dataset at specified indices with given transform.
    Same as torch.utils.data.Subset but with transform support.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        transform (Transform): Transform operation to be applied to the dataset.
    """

    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)
