import tensorflow as tf
from model import logDcoral
from solver import Solver

flags = tf.compat.v1.app.flags
flags.DEFINE_string('mode', 'train', "'train', or 'test'")
flags.DEFINE_string('method', 'baseline',
                    "the regularizer: 'baseline' (no regularizer), 'd-coral', 'log-d-coral' or 'entropy'")
flags.DEFINE_string('model_save_path', 'model', "base directory for saving the models")
flags.DEFINE_string('device', '/gpu:0', "/gpu:id number")
flags.DEFINE_string('alpha', '1.', "regularizer weight")
flags.DEFINE_string('sid', '1', 'source id')
flags.DEFINE_string('tid', '2', 'target id')
FLAGS = flags.FLAGS

def main(argv=None):
    tf.compat.v1.disable_eager_execution()
    with tf.device(FLAGS.device):
        model_save_path = FLAGS.model_save_path + '/' + FLAGS.method + '/alpha_' + FLAGS.alpha
        log_dir = 'logs/' + FLAGS.method + '/alpha_' + FLAGS.alpha
        res_dir = 'result/' + FLAGS.method + '/alpha_' + FLAGS.alpha
        model = logDcoral(mode=FLAGS.mode, method=FLAGS.method, hidden_size=16, learning_rate=0.0005,
                          alpha=float(FLAGS.alpha))
        solver = Solver(model, batch_size=128, model_save_path=model_save_path, log_dir=log_dir, res_dir=res_dir,
                        sid=int(FLAGS.sid), tid=int(FLAGS.tid))

        # Create directory if it does not exist
        if not tf.io.gfile.exists(model_save_path): tf.io.gfile.makedirs(model_save_path)

        if FLAGS.mode == 'train': solver.train()
        elif FLAGS.mode == 'test': solver.test()
        elif FLAGS.mode == 'tsne': solver.tsne()
        else: print('Unrecognized mode. Current recognizable modes are: train, test, tsne.')

if __name__ == '__main__':
    tf.compat.v1.app.run()
