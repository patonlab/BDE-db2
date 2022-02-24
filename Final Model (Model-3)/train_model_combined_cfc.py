import os

import numpy as np
import tensorflow as tf
from pathlib import Path

import pandas as pd

import rdkit.Chem  # noqa: F401 isort:skip
import nfp  # isort:skip
import tensorflow_addons as tfa  # isort:skip

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras import layers

from preprocess_inputs_cfc import preprocessor
preprocessor.from_json('20220221_tfrecords_multi_halo_cfc/preprocessor.json')

data_dir = "/home/svss/projects/Project-BDE/20220221-new-models/"
data = pd.read_pickle(Path(data_dir, "20220221_tfrecords_multi_halo_cfc/model_inputs.p"))
train = data[data.set == "train"]
valid = data[data.set == "valid"]

batch_size = 128

output_signature = {
    **preprocessor.output_signature,
    **{"output": tf.TensorSpec(shape=(None, 2), dtype=tf.float32)},
}

padding_values = {
    **preprocessor.padding_values,
    **{"output": tf.constant(np.nan, dtype=tf.float32)},
}


@tf.function
def split_output(input_dict):
    copied_dict = dict(input_dict)
    output = copied_dict.pop("output")
    return copied_dict, output


train_dataset = (
    tf.data.Dataset.from_generator(
        lambda: iter(train.model_inputs), output_signature=output_signature,
    )
    .cache()
    .shuffle(buffer_size=len(train))
    .padded_batch(batch_size=batch_size, padding_values=padding_values)
    .map(split_output)
    .prefetch(tf.data.experimental.AUTOTUNE)
)

valid_dataset = (
    tf.data.Dataset.from_generator(
        lambda: iter(valid.model_inputs), output_signature=output_signature,
    )
    .cache()
    .padded_batch(batch_size=batch_size, padding_values=padding_values)
    .map(split_output)
    .prefetch(tf.data.experimental.AUTOTUNE)
)

batch_size = 128
atom_features = 128
num_messages = 6

class Slice(layers.Layer):
    def call(self, inputs):
        input_shape = tf.shape(inputs)
        num_bonds = input_shape[1] / 2
        output = tf.slice(inputs, [0, 0, 0], [-1, num_bonds, -1])
        output.set_shape(self.compute_output_shape(inputs.shape))
        return output

    def compute_output_shape(self, input_shape):
        return [input_shape[0], None, input_shape[2]]

# Define keras model
bond_indices = layers.Input(shape=[None], dtype=tf.int64, name='bond_indices')
atom_class = layers.Input(shape=[None], dtype=tf.int64, name='atom')
bond_class = layers.Input(shape=[None], dtype=tf.int64, name='bond')
connectivity = layers.Input(shape=[None, 2], dtype=tf.int64, name='connectivity')

input_tensors = [bond_indices, atom_class, bond_class, connectivity]

# Initialize the atom states
atom_state = layers.Embedding(preprocessor.atom_classes, atom_features,
                              name='atom_embedding', mask_zero=True)(atom_class)

# Initialize the bond states
bond_state = layers.Embedding(preprocessor.bond_classes, atom_features,
                              name='bond_embedding', mask_zero=True)(bond_class)

# # Initialize the bond states
# bde_mean = layers.Embedding(preprocessor.bond_classes, 1,
#                              name='bde_mean', mask_zero=True)(bond_class)

# bdfe_mean = layers.Embedding(preprocessor.bond_classes, 1,
#                              name='bdfe_mean', mask_zero=True)(bond_class)

# Initialize the bond states
bde_mean = layers.Embedding(
    preprocessor.bond_classes, 2, name="bde_mean", mask_zero=True
)(bond_class)

for _ in range(num_messages):  # Do the message passing
    new_bond_state = nfp.EdgeUpdate()([atom_state, bond_state, connectivity])
    bond_state = layers.Add()([bond_state, new_bond_state])
    new_atom_state = nfp.NodeUpdate()([atom_state, bond_state, connectivity])    
    atom_state = layers.Add()([atom_state, new_atom_state])   

bond_state = nfp.Reduce(reduction="mean", name="bond_state_reduced")(
        [bond_state, bond_indices, bond_state]
    )
bond_state = Slice()(bond_state)

bde_pred = layers.Dense(2, name="bde_no_mean", use_bias=False)(bond_state)

bde_mean = nfp.Reduce(reduction="mean")([bde_mean, bond_indices, bde_mean])
bde_mean = Slice()(bde_mean)

bde_pred = layers.Add(name="bde")([bde_pred, bde_mean])

model = tf.keras.Model(input_tensors, [bde_pred])
    
# bond_state = nfp.Reduce(reduction='mean')([bond_state, bond_indices, bond_state])
# bde_mean = nfp.Reduce(reduction='mean')([bde_mean, bond_indices, bde_mean])
# bdfe_mean = nfp.Reduce(reduction='mean')([bdfe_mean, bond_indices, bdfe_mean])

# bde_pred = layers.Dense(1)(bond_state)
# bde_pred = layers.Add(name='bde')([bde_pred, bde_mean])

# bdfe_pred = layers.Dense(1)(bond_state)
# bdfe_pred = layers.Add(name='bdfe')([bdfe_pred, bdfe_mean])

# model = tf.keras.Model(input_tensors, [bde_pred, bdfe_pred])

learning_rate = tf.keras.optimizers.schedules.InverseTimeDecay(1E-3, 1, 1E-5)
weight_decay  = tf.keras.optimizers.schedules.InverseTimeDecay(1E-5, 1, 1E-5)
optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
model.compile(loss=nfp.masked_mean_absolute_error, optimizer=optimizer)

model_name = '20220221_model_multi_halo_cfc'

if not os.path.exists(model_name):
    os.makedirs(model_name)

filepath = model_name + "/best_model.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, save_best_only=True, verbose=0)
csv_logger = tf.keras.callbacks.CSVLogger(model_name + '/log.csv')

model.fit(train_dataset,
          validation_data=valid_dataset,
          epochs=500,
          callbacks=[checkpoint, csv_logger],
          verbose=2)