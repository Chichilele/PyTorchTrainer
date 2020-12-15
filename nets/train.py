import torch
import mlflow
import mlflow.pytorch


class Trainer:
    """Trainer class for Pytorch modules. Implements the training phase.

    Attributes:
        net (nn.Module) : neural network to be trained.
        optimizer : optimizer to use for training.
        criterion : loss function.
        train_loader : training data loader.
        test_loader : test data loader.
        log_interval : interval between each log (both printed and stored for plotting).
        lr_scheduer : learning rate scheduler.

    """

    def __init__(
        self,
        net,
        optimizer,
        criterion,
        train_loader,
        test_loader,
        log_interval=None,
        lr_scheduler=None,
        modelname=None,
    ):
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.log_interval = (
            len(train_loader) // log_interval
            if log_interval
            else len(train_loader) // 10
        )
        self.lr_scheduler = lr_scheduler
        self.modelname = modelname if modelname else "pytorch_model"

        self._trainset_size = len(self.train_loader.dataset)
        self._trainloader_size = len(self.train_loader)

        self._testset_size = len(self.test_loader.dataset)
        self._testloader_size = len(self.test_loader)

        self.step = 0

    def log_epoch(
        self, epoch, batch_idx, img_done, percentage_done, cum_loss, cum_accuracy
    ):
        avg_loss = sum(cum_loss) / len(cum_loss)
        avg_accuracy = sum(cum_accuracy) / len(cum_accuracy)
        log_message = f"Train Epoch: {epoch} [{img_done:5}/{self._trainset_size} ({percentage_done:2.0f}%)]\tLoss: {avg_loss:.6f}\tAccuracy: {100.0 * avg_accuracy:2.0f}%"
        print(log_message)
        mlflow.log_metric("train_loss", avg_loss)
        mlflow.log_metric("train_accuracy", avg_accuracy)
        mlflow.log_metric("train_loss", avg_loss, step=self.step)
        mlflow.log_metric("train_accuracy", avg_accuracy, step=self.step)


    def train_step(self, epoch):
        """Trains one step (epoch) and logs.

        Args:
            epoch (int): epoch id to be processed (for logging purpose only).

        """
        self.net.train()

        cum_loss, cum_accuracy = [], []
        for batch_idx, (data, target) in enumerate(self.train_loader, 1):
            data, target = data.cuda(), target.cuda()
            self.optimizer.zero_grad()
            output = self.net(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            ## log
            cum_loss += [loss.item()]
            pred = output.data.max(1, keepdim=True)[1]
            cum_accuracy += [
                pred.eq(target.data.view_as(pred)).sum().item() / len(target)
            ]

            if batch_idx % self.log_interval == 0:
                img_done = batch_idx * self.train_loader.batch_size
                percentage_done = 100.0 * batch_idx / self._trainloader_size
                self.log_epoch(
                    epoch, batch_idx, img_done, percentage_done, cum_loss, cum_accuracy
                )
                cum_loss = []

            self.step += 1


        ## after end of train loop
        if cum_loss:
            img_done = self._trainset_size
            percentage_done = 100.0
            self.log_epoch(
                epoch, batch_idx, img_done, percentage_done, cum_loss, cum_accuracy
            )

    def test(self):
        """Test function evaluating the training set.

        Evaluates network against test set andlogs results to `test_loss` and `test_accuracy` in MLflow metrics.

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
        test_loss /= self._testloader_size

        test_accuracy = correct.item() / self._testset_size
        print(
            f"\nTest set: Avg. loss: {test_loss:.4f}, Accuracy: {correct}/{self._testset_size} ({100.0 * test_accuracy:2.0f}%)\n"
        )

        # Log to MLflow
        mlflow.log_metric("test_loss", test_loss, step=self.step)
        mlflow.log_metric("test_accuracy", test_accuracy, step=self.step)

    def train(self, n_epochs):
        """Neural network training routine.

        Args:
            n_epochs (int): number of epochs to trained the network on.

        """

        self.test()
        for epoch in range(1, n_epochs + 1):
            self.train_step(epoch)
            self.test()
            mlflow.pytorch.log_model(self.net, f"{self.modelname}/epochs/{epoch}")

            if self.lr_scheduler:
                self.lr_scheduler.step()

        mlflow.pytorch.log_model(self.net, self.modelname)
