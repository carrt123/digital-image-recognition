import matplotlib.pyplot as plt


def draw_train_process(iters, train_loss, train_accs):
    title = "training loss/training accs"
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=14)
    plt.ylabel("loss/acc", fontsize=14)
    plt.plot(iters, train_loss, color='red', label='training loss')
    plt.plot(iters, train_accs, color='green', label='training accs')
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig('./mnist/accuracy.png')