import argparse
import tensorflow as tf
from tensorflow.python.keras import backend as K
from simclr_models import SelectLayer

parser = argparse.ArgumentParser("flops_parser")
parser.add_argument("model_path", type=str, help="Path to the .keras file of the model to calculate the floating point operations of.")
args = parser.parse_args()

with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    K.set_session(sess)
    model = tf.keras.models.load_model(args.model_path)
    model.summary()
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)
    print(f"Model FLOPs: {flops.total_float_ops}")


