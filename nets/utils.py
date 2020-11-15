import matplotlib.pyplot as plt


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
