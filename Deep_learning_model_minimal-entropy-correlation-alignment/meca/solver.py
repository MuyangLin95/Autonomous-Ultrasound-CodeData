import tensorflow as tf
import tf_slim as slim
import numpy as np
import pickle
import os
import scipy.io
import time
import matplotlib.pyplot as plt

import utils
from sklearn.manifold import TSNE

class Solver(object):
    def __init__(self, model, batch_size=8, train_iter=200,
                 source_dir='source', target_dir='target', log_dir='logs', res_dir='result',
                 model_save_path='model', sid=1, tid=2):
        self.model = model
        self.batch_size = batch_size
        self.train_iter = train_iter
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.log_dir = log_dir
        self.res_dir = res_dir
        self.model_save_path = model_save_path
        self.trained_model = model_save_path + '/model'
        self.config = tf.compat.v1.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.acc_curve = []
        self.loss_curve = []
        self.sid = sid
        self.tid = tid

    def load_dataset(self, image_dir, dataset='commercial', split='train'):
        assert (dataset in ['commercial', 'chw', 'mnist', 'svhn'])
        if dataset in ['commercial', 'chw']:
            id = self.sid if dataset == 'commercial' else self.tid
            datatext = 'Commercial' if dataset == 'commercial' else 'CHW'
            print('Loading ' + datatext + ' dataset.')
            if (split == 'train'):
                images = np.load(f"tds/{id}.npy")
                labels = np.load(f"tds/{id}_Y.npy")
            else:
                images = np.load(f"tds/va_{id}.npy")
                labels = np.load(f"tds/va_{id}_Y.npy")

        elif dataset == 'mnist':
            print('Loading ' + dataset.upper() + ' dataset.')
            image_file = 'train.pkl' if split == 'train' else 'test.pkl'
            image_dir = os.path.join(image_dir, image_file)
            with open(image_dir, 'rb') as f:
                mnist = pickle.load(f)
            images = mnist['X'] / 127.5 - 1
            labels = mnist['y']
            labels = np.squeeze(labels).astype(int)

        elif dataset == 'svhn':
            print('Loading SVHN dataset.')
            image_file = 'train_32x32.mat' if split == 'train' else 'test_32x32.mat'
            image_dir = os.path.join(image_dir, image_file)
            svhn = scipy.io.loadmat(image_dir)
            images = np.transpose(svhn['X'], [3, 0, 1, 2]) / 127.5 - 1
            labels = svhn['y'].reshape(-1)
            labels[np.where(labels == 10)] = 0

        return images, labels

    def train(self):
        # Makes the log directory if it doesn't exist.
        if tf.io.gfile.exists(self.log_dir): tf.io.gfile.rmtree(self.log_dir)
        tf.io.gfile.makedirs(self.log_dir)

        # Makes the result directory if it doesn't exist.
        if tf.io.gfile.exists(self.res_dir): tf.io.gfile.rmtree(self.res_dir)
        tf.io.gfile.makedirs(self.res_dir)

        print('Training.')
        trg_images, trg_labels = self.load_dataset(self.target_dir, dataset='chw', split='train')
        trg_test_images, trg_test_labels = self.load_dataset(self.target_dir, dataset='chw', split='test')

        src_images, src_labels = self.load_dataset(self.source_dir, dataset='commercial', split='train')
        src_test_images, src_test_labels = self.load_dataset(self.source_dir, dataset='commercial', split='test')

        print("***************")
        print(src_images.shape, trg_images.shape)
        print("***************")

        # build a graph
        model = self.model
        model.build_model()

        config = tf.compat.v1.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True

        with tf.compat.v1.Session(config=config) as sess:
            tf.compat.v1.global_variables_initializer().run()
            saver = tf.compat.v1.train.Saver()

            summary_writer = tf.compat.v1.summary.FileWriter(logdir=self.log_dir,
                                                             graph=tf.compat.v1.get_default_graph())

            print('Start training.')
            trg_count = 0
            t = 0
            start_time = time.time()

            for step in range(self.train_iter):
                trg_count += 1
                t += 1

                i = step % int(src_images.shape[0] / self.batch_size)
                j = step % int(trg_images.shape[0] / self.batch_size)

                feed_dict = {model.src_images: src_images[i * self.batch_size:(i + 1) * self.batch_size],
                             model.src_labels: src_labels[i * self.batch_size:(i + 1) * self.batch_size],
                             model.trg_images: trg_images[j * self.batch_size:(j + 1) * self.batch_size],
                             model.trg_test_images: trg_test_images[:],
                             model.trg_test_labels: trg_test_labels[:],
                             }

                sess.run(model.train_op, feed_dict)

                if (t % 1 == 0 or t == 1):
                    summary, l_c, l_d, src_acc, tar_acc, tar_loss = sess.run(
                        [model.summary_op, model.class_loss, model.domain_loss, model.src_accuracy, model.trg_accuracy,
                         model.trg_class_loss], feed_dict)
                    summary_writer.add_summary(summary, t)
                    print(
                        'Step: [%d/%d]  c_loss: [%.6f]  d_loss: [%.6f]  train acc: [%.2f] val acc: [%.3f] val loss: [%.3f]' \
                        % (t, self.train_iter, l_c, l_d, src_acc, tar_acc, tar_loss))
                    self.acc_curve.append(tar_acc)
                    self.loss_curve.append(l_d)

            with open(self.res_dir + '/' + model.method + '/' + str(model.alpha) + '/time_' + str(model.alpha)
                      + '_' + model.method + '.txt', "a") as resfile:
                resfile.write(str((time.time() - start_time) / float(self.train_iter)) + '\n')
                saver.save(sess, os.path.join(self.model_save_path, 'model'))

        with open(self.res_dir + "/cross_result.txt", "a") as f:
            print(self.sid, self.tid, np.max(self.acc_curve[-200:]), file=f)
            print(self.acc_curve, file=f)
            print(self.loss_curve, file=f)


    def test(self):
        trg_images, trg_labels = self.load_dataset(self.target_dir, dataset='chw', split='test')
        print(trg_images.shape)
        # Build a graph
        model = self.model
        model.build_model()

        config = tf.compat.v1.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True

        with tf.compat.v1.Session(config=config) as sess:
            tf.compat.v1.global_variables_initializer().run()

            print('Loading model.')
            variables_to_restore = slim.get_model_variables()
            restorer = tf.compat.v1.train.Saver(variables_to_restore)
            restorer.restore(sess, self.trained_model)

            trg_acc, trg_entr, trg_res = sess.run(fetches=[model.trg_accuracy, model.trg_entropy, model.trg_softmax],
                                                  feed_dict={model.trg_images: trg_images[:],
                                                             model.trg_labels: trg_labels[:]})

            print('Test Accuracy: [%.3f]' % (trg_acc))
            print('Entropy: [%.3f]' % (trg_entr))

            print(trg_res)
            print(trg_labels)

            with open(self.res_dir + '/' + model.method + '/' + str(model.alpha) +
                      'test_' + str(model.alpha) + '_' + model.method + '.txt', "a") as resfile:
                resfile.write(str(trg_acc) + '\t' + str(trg_entr) + '\n')

    def tsne(self, n_samples=60):

        source_images, source_labels = self.load_dataset(self.source_dir, dataset='commercial', split='test')
        target_images, target_labels = self.load_dataset(self.target_dir, dataset='chw', split='test')

        model = self.model
        model.build_model()

        config = tf.compat.v1.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True

        with tf.compat.v1.Session(config=config) as sess:
            print('Loading test model.')
            variables_to_restore = tf.compat.v1.global_variables()
            restorer = tf.compat.v1.train.Saver(variables_to_restore)
            restorer.restore(sess, self.trained_model)

            target_images = target_images[:n_samples]
            target_labels = target_labels[:n_samples]
            source_images = source_images[n_samples * 5:n_samples * 6]
            source_labels = source_labels[n_samples * 5:n_samples * 6]
            print(source_labels.shape)

            assert len(target_labels) == len(source_labels)

            src_labels = utils.one_hot(source_labels.astype(int), 2)
            trg_labels = utils.one_hot(target_labels.astype(int), 2)

            n_slices = 32

            fx_src = np.empty((0, model.hidden_repr_size))
            fx_trg = np.empty((0, model.hidden_repr_size))

            for src_im, trg_im in zip(np.array_split(source_images, n_slices),
                                      np.array_split(target_images, n_slices),
                                      ):
                feed_dict = {model.src_images: src_im[:], model.trg_images: trg_im[:]}

                fx_src_, fx_trg_ = sess.run([model.src_hidden, model.trg_hidden], feed_dict)

                fx_src = np.vstack((fx_src, np.squeeze(fx_src_)))
                fx_trg = np.vstack((fx_trg, np.squeeze(fx_trg_)))

            assert len(src_labels) == len(fx_src)
            assert len(trg_labels) == len(fx_trg)

            print('Computing T-SNE.')

            model = TSNE(n_components=2, random_state=42)

            import matplotlib
            matplotlib.rcParams['savefig.dpi'] = 300

            TSNE_hA = model.fit_transform(np.vstack((fx_src, fx_trg)))
            plt.scatter(TSNE_hA[:, 0], TSNE_hA[:, 1],
                        c=np.hstack((np.argmax(src_labels, 1), np.argmax(trg_labels, 1),)), s=10)
            plt.axis("off")
            plt.show()

            TSNE_org = model.fit_transform(
                np.vstack((target_images.reshape(60, 2500), source_images.reshape(60, 2500))))
            plt.scatter(TSNE_org[:, 0], TSNE_org[:, 1],
                        c=np.hstack((np.argmax(src_labels, 1), np.argmax(trg_labels, 1),)), s=10)
            plt.axis("off")
            plt.show()

            print(TSNE_hA.tolist())
            print(TSNE_org.tolist())

if __name__ == '__main__':
    print('This script is only for the Solver class: To utilize this class, run train.py or main.py.')
