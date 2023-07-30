
import os
import pprint
import tempfile
from typing import Dict, Text
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
import random
import logging
import pandas as pd
import tensorflow_ranking as tfr
from tqdm import tqdm



DATA_NAME_sessions = "train_sessions.csv"
DATA_NAME_purchases = "train_purchases.csv"
def load_data(DATA_NAME_sessions, DATA_NAME_purchases):
    
    train_sessions = pd.read_csv(DATA_NAME_sessions,parse_dates=['date'],dtype={
                     'session_id': int,
                     'item_id': int
                 })
    train_sessions['timestemp'] = train_sessions['date'].values.astype('datetime64[s]').astype(np.int64) # to_unix/to_timestemp
    # Drop columns, group by session_id, and pad with 0 - length = max length
    train_sessions = train_sessions[['session_id', 'item_id','timestemp' ]]
    unique_item_ids = np.unique(train_sessions['item_id'])
    train_sessions = train_sessions[['session_id', 'item_id','timestemp' ]].groupby('session_id').agg(list).reset_index(level=0)
    # Pad with 0
    max_len = max([len(line) for line in train_sessions.item_id])
    [ line.extend([0]*(max_len-len(line))) for line in train_sessions.item_id if len(line)<max_len]
    [ line.extend([0]*(max_len-len(line))) for line in train_sessions.timestemp if len(line)<max_len]


    # Load Purchase data
    col_names=['session_id_p', 'item_id_p', 'date_p']
    train_purchases = pd.read_csv(DATA_NAME_purchases,parse_dates=['date'],dtype={
                         'session_id': int,
                         'item_id': int
                     })
    train_purchases['timestemp'] = train_purchases['date'].values.astype('datetime64[s]').astype(np.int64) # to_unix/to_timestemp
    unique_item_ids_p = np.unique(train_purchases['item_id'])

    train_df = train_sessions.merge(train_purchases, on='session_id', how='inner', suffixes=('_s', '_p'))
    train_df.drop('date',axis = 1, inplace  = True)
    train_df.rename(columns = { 'item_id_s':'item_id', 'timestemp_s': 'timestemp', 'item_id_p':'label'}, inplace = True)
    train_df.drop('timestemp_p',axis = 1, inplace  = True)
    
    unique_item_ids_p = np.unique(train_purchases['item_id']).tolist()
    unique_item_ids = unique_item_ids.tolist()
    unique_item_ids.extend(unique_item_ids_p)
    unique_item_ids = list(set(unique_item_ids))
    unique_item_ids = np.array(unique_item_ids).astype(str)
    
    return unique_item_ids,train_df


def create_examples(train_df):
    examples = []
    for index,row in tqdm(train_df.iterrows()):
        item_id = [x for x in row['item_id']]
        timestemp = [x for x in row['timestemp']]
        label = row['label']
        feature = {"context_item_id":
               tf.train.Feature(
                   int64_list=tf.train.Int64List(value= item_id)),
               "context_item_timestemp":
               tf.train.Feature(
                   int64_list=tf.train.Int64List(value= timestemp)),
                   "label":
               tf.train.Feature(
                   int64_list=tf.train.Int64List(value= [label])),
                }
        tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
        examples.append(tf_example)
    return examples


OUTPUT_TRAINING_DATA_FILENAME = 'train.tfrecord'
OUTPUT_TESTING_DATA_FILENAME = 'test.tfrecord'

def generate_train_test(examples, train_data_fraction=0.9,random_seed = 123, shuffle = True):
    if shuffle:
        random.seed(random_seed)
        random.shuffle(examples)
    last_train_index = round(len(examples) * train_data_fraction)
    train_examples = examples[:last_train_index]
    test_examples = examples[last_train_index:]
    return train_examples, test_examples



def write_tfrecords(tf_examples, filename):
  """Writes tf examples to tfrecord file, and returns the count."""
  with tf.io.TFRecordWriter(filename) as file_writer:
    length = len(tf_examples)
    progress_bar = tf.keras.utils.Progbar(length)
    for example in tf_examples:
      file_writer.write(example.SerializeToString())
      progress_bar.add(1)
    return length



def generate_datasets(examples, 
                      random_seed = 123,
                      shuffle = True,
                      train_data_fraction=0.9,
                      train_filename=OUTPUT_TRAINING_DATA_FILENAME,
                      test_filename=OUTPUT_TESTING_DATA_FILENAME
                     ):
    
    train_examples, test_examples = generate_train_test(examples,
                                                        train_data_fraction=train_data_fraction,
                                                        random_seed = random_seed,
                                                        shuffle = shuffle)

    logging.info("Writing generated training examples.")
    train_file = train_filename
    train_size = write_tfrecords(tf_examples=train_examples, filename=train_file)
    logging.info("Writing generated testing examples.")
    test_file = test_filename
    test_size = write_tfrecords(tf_examples=test_examples, filename=test_file)
    stats = {
          "train_size": train_size,
          "test_size": test_size,
          "train_file": train_file,
          "test_file": test_file,
      }
        
    return stats

stats = generate_datasets(examples)
logging.info("Generated dataset: %s", stats)


train_filename = "train.tfrecord"
train = tf.data.TFRecordDataset(train_filename)

test_filename = "test.tfrecord"
test = tf.data.TFRecordDataset(test_filename)


# In[41]:


feature_description = {
    'context_item_id': tf.io.FixedLenFeature([100], tf.int64, default_value=np.repeat(0, 100)),
    'context_item_timestemp': tf.io.FixedLenFeature([100], tf.int64, default_value=np.repeat(0, 100)),
    'label': tf.io.FixedLenFeature([1], tf.int64, default_value=0),
}

def _parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)

train_ds = train.map(_parse_function).map(lambda x: {
    "context_item_id": tf.strings.as_string(x["context_item_id"]),
    #"context_item_timestemp": (x["context_item_timestemp"]),
    "label": tf.strings.as_string(x["label"])
})

test_ds = test.map(_parse_function).map(lambda x: {
    "context_item_id": tf.strings.as_string(x["context_item_id"]),
    #"context_item_timestemp": (x["context_item_timestemp"]),
    "label": tf.strings.as_string(x["label"])
})

embedding_dimension = 32

query_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(
      vocabulary=unique_item_ids, mask_token=None),
    tf.keras.layers.Embedding(len(unique_item_ids) + 1, embedding_dimension), 
    tf.keras.layers.GRU(embedding_dimension)
])

candidate_model = tf.keras.Sequential([
  tf.keras.layers.StringLookup(
      vocabulary=unique_item_ids, mask_token=None),
  tf.keras.layers.Embedding(len(unique_item_ids) + 1, embedding_dimension)
])


task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
    candidates=item_ids.batch(128).map(candidate_model),
    metrics=[
        tfr.keras.metrics.MRRMetric(),
        tf.keras.metrics.TopKCategoricalAccuracy()
    ],
  )
)


item_ids = tf.data.Dataset.from_tensor_slices(unique_item_ids)


class Model(tfrs.Model):

    def __init__(self, query_model, candidate_model):
        super().__init__()
        self._query_model = query_model
        self._candidate_model = candidate_model

        self._task = task

    def compute_loss(self, features, training=False):
        watch_history = features["context_item_id"]
        watch_next_label = features["label"]

        query_embedding = self._query_model(watch_history)       
        candidate_embedding = self._candidate_model(watch_next_label)

        return self._task(query_embedding, candidate_embedding, compute_metrics = not training)

model = Model(query_model, candidate_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

cached_train = train_ds.batch(4096).cache()
cached_test = test_ds.batch(512).cache()

model.fit(cached_train, epochs=3)
model.evaluate(cached_test, return_dict=True)
