from ..model_building import model_build_conv, model_build_dense
from keras.utils import plot_model
import os
import sys

this_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
os.chdir(this_dir)
print(this_dir)
conv_model = model_build_conv(is_only_for_plotting=True)
dense_model = model_build_dense(is_only_for_plotting=True)

plot_model(conv_model, show_shapes=True,
           to_file='../../docs/images/conv_model.png')
plot_model(dense_model, show_shapes=True,
           to_file='../../docs/images/dense_model.png')
