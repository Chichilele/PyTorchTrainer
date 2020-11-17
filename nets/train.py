import torch


class Trainer:
    """Trainer class for Pytorch modules. Implements the training phase.

    Attributes:
        net (nn.Module) : neural network to be trained.
        optimizer : optimizer to use for training.
        criterion : loss function.
        train_loader : training data loader.
        test_loader : test data loader.
        log_interval : interval between each log (both printed and stored for plotting).
        lr_scheduer : learning rate scheduler

    """

    def __init__(
        self, net, optimizer, criterion, train_loader, test_loader, log_interval=100, lr_scheduler=None
    ):
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.log_interval = log_interval
        self.lr_scheduler = lr_scheduler

        self.train_losses = []
        self.train_counter = []
        self.test_losses = []
        self.test_counter = []

    def train_step(self, epoch):
        """Trains one step (epoch) and logs.

        Args:
            epoch (int): epoch id to be processed (for logging purpose only).

        """
        self.net.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.cuda(), target.cuda()
            self.optimizer.zero_grad()
            output = self.net(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                img_done = batch_idx * len(data)
                dataset_size = len(self.train_loader.dataset)
                percentage_done = 100.0 * batch_idx / len(self.train_loader)
                print(
                    f"Train Epoch: {epoch} [{img_done}/{dataset_size} ({percentage_done:.0f}%)]\tLoss: {loss.item():.6f}"
                )

                self.train_losses.append(loss.item())
                self.train_counter.append(
                    (batch_idx * self.train_loader.batch_size)
                    + ((epoch - 1) * dataset_size)
                )
                ## save model
                torch.save(self.net.state_dict(), "./models/nextjournal_model.pth")
                torch.save(
                    self.optimizer.state_dict(),
                    "./models/nextjournal_self.optimizer.pth",
                )

    def test(self):
        """test function evaluating the training set.

        logs results to `test_losses` and prints results.

        """
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.cuda()
                target = target.cuda()
                output = self.net(data)
                test_loss += self.criterion(output, target).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(self.test_loader)
        self.test_losses.append(test_loss)

        accuracy_rate = 100.0 * correct / len(self.test_loader.dataset)
        print(
            f"\nTest set: Avg. loss: {test_loss:.4f}, Accuracy: {correct}/{len(self.test_loader.dataset)} ({accuracy_rate:.0f}%)\n"
        )

    def train(self, n_epochs):
        """Neural network training routine.

        Args:
            n_epochs (int): number of epochs to trained the network on.

        Returns:
            dict: dict containing train and test losses.

        """
        self.test_counter = [
            i * len(self.train_loader.dataset)
            for i in range(len(self.test_counter) + n_epochs + 1)
        ]
        self.test()
        for epoch in range(1, n_epochs + 1):
            self.train_step(epoch)
            self.test()
            if self.lr_scheduler: self.lr_scheduler.step()

        results = {
            "train_loss": (self.train_losses, self.train_counter),
            "test_loss": (self.test_losses, self.test_counter),
        }

        return self.net, results
