import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .02  #Edited for 3c
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False
    use_relu = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    configs = [
        ("Improved weights", False, True, False),
        ("Improved sigmoid", True, True, False),
        ("Momentum", True, True, True),
    ]

    for name, use_improved_sigmoid, use_improved_weight_init, use_momentum in configs:
        model = SoftmaxModel(
            neurons_per_layer,
            use_improved_sigmoid,
            use_improved_weight_init,
            use_relu)
        trainer = SoftmaxTrainer(
            momentum_gamma, use_momentum,
            model, learning_rate, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        )

        train_history, val_history = trainer.train(num_epochs)

        # Plot loss
        plt.subplot(1, 2, 1)
        utils.plot_loss(train_history["loss"], "Training loss - " + name, npoints_to_average=10)
        utils.plot_loss(val_history["loss"], "Validation loss - " + name, npoints_to_average=10)
        plt.ylabel("Cross Entropy Loss")
        plt.xlabel("Number of Training Steps")
        plt.ylim([0.0, .5])
        #plt.legend()
        plt.grid(True)
        

        plt.subplot(1, 2, 2)
        utils.plot_loss(train_history["accuracy"], "Training accuracy - " + name)
        utils.plot_loss(val_history["accuracy"], "Validation accuracy - " + name)
        plt.ylabel("Accuracy")
        plt.xlabel("Number of Training Steps")
        plt.ylim([0.85, 1.0])
        plt.legend()
        plt.grid(True)
    
    plt.savefig("task3_model_comparison.png")
    plt.show()




if __name__ == "__main__":
    main()
