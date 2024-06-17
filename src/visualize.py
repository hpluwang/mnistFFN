import matplotlib.pyplot as plt

def plot_metrics(losses, trainAcc, testAcc, save_path=None):
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))

    ax[0].plot(losses)
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_ylim([0, 3])
    ax[0].set_title('Model loss')

    ax[1].plot(trainAcc, label='Train')
    ax[1].plot(testAcc, label='Test')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy (%)')
    ax[1].set_ylim([10, 100])
    ax[1].set_title(f'Final model test accuracy: {testAcc[-1]:.2f}%')
    ax[1].legend()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
