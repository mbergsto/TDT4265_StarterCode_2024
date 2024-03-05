import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy


class ExampleModel(nn.Module):
    def __init__(self, image_channels, num_classes):
        """
        Is called when model is initialized.
        Args:
            image_channels. Number of color channels in image (3)
            num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        self.num_classes = num_classes
        kernel_size_conv = 3
        kernel_size_else = 2
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            #Layer 1 - first convolutional layer
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=128,
                kernel_size = kernel_size_conv,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout2d(p=0.05),
            nn.MaxPool2d(
                kernel_size=kernel_size_else, 
                stride=2),
            
            #Layer 2 - second convolutional layer
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size = kernel_size_conv,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout2d(p=0.05),
            nn.MaxPool2d(
                kernel_size=kernel_size_else, 
                stride=2),

            #Layer 3 - third convolutional layer
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size = kernel_size_conv,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512),
            nn.Dropout2d(p=0.05),
            nn.MaxPool2d(
                kernel_size=kernel_size_else, 
                stride=2),
            
            #Layer 4 - third convolutional layer
            nn.Conv2d(
                in_channels=512,
                out_channels=1024,
                kernel_size = kernel_size_conv,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024),
            nn.Dropout2d(p=0.05),
            nn.MaxPool2d(
                kernel_size=kernel_size_else, 
                stride=2),
        )      

        
    
        # Calculate the number of output features
        self.num_output_features = 1024 * 2 * 2

        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Flatten(),
            #Layer 5 - first fully connected layer
            nn.Linear(self.num_output_features, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.05),
            #Layer 6 - second fully connected layer
            nn.Linear(512, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.05),
            #Layer 7 - third fully connected layer
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        batch_size = x.shape[0]
        out = x
        out = self.feature_extractor(out).view(batch_size, -1)
        out = self.classifier(out)
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (
            batch_size,
            self.num_classes,
        ), f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(
        trainer.train_history["loss"], label="Training loss", npoints_to_average=10
    )
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()




def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result!
    utils.set_seed(0)
    print(f"Using device: {utils.get_device()}")
    epochs = 10
    batch_size = 64
    learning_rate = 5e-2
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    model = ExampleModel(image_channels=3, num_classes=10)
    trainer = Trainer(
        batch_size, learning_rate, early_stop_count, epochs, model, dataloaders
    )
    trainer.train()
    #create_plots(trainer, "task3")

    train_loss, train_accuracy = compute_loss_and_accuracy(trainer.dataloader_train, model, nn.CrossEntropyLoss())
    validation_loss, validation_accuracy = compute_loss_and_accuracy(trainer.dataloader_val, model, nn.CrossEntropyLoss())
    test_loss, test_accuracy = compute_loss_and_accuracy(trainer.dataloader_test, model, nn.CrossEntropyLoss())

    print(f"Train accuracy: {train_accuracy:.3f}, Validation accuracy: {validation_accuracy:.3f}, Test accuracy: {test_accuracy:.3f}")
    print(f"Train loss: {train_loss:.3f}, Validation loss: {validation_loss:.3f}, Test loss: {test_loss:.3f}")


if __name__ == "__main__":
    main()
