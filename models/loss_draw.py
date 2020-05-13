import matplotlib.pyplot as plt
import tensorflow as tf


# save loss, acc
class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], lw=1.5, color='r', label='train acc', marker='.', markevery=2,
                 mew=1.5)
        # loss
        plt.plot(iters, self.losses[loss_type], lw=1.5, color='g', label='train loss', marker='.', markevery=2, mew=1.5)
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], lw=1.5, color='b', label='val acc', marker='.', markevery=2,
                     mew=1.5)
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], lw=1.5, color='darkorange', label='val loss', marker='.',
                     markevery=2, mew=1.5)
        plt.grid(True)
        plt.xlim(-0.1, 10)
        plt.ylim(-0.01, 1.01)
        plt.xlabel(loss_type)
        plt.ylabel('ACC-LOSS')
        plt.legend(loc="center right")
        plt.savefig("acc_loss.pdf")
        plt.show()
