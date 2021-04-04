import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier

import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import Normalization

def dataframe_to_dataset(dataframe, features):
    dataframe = dataframe.copy()
    labels = dataframe.pop('label')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe[features]), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature

def predict_nn(train_df, test_df, feature_combinations):
    for features in feature_combinations:
        print(features)
        train_ds = dataframe_to_dataset(train_df, features)
        test_ds = dataframe_to_dataset(test_df, features)
        train_ds = train_ds.batch(32)
        test_ds = test_ds.batch(32)

        all_inputs = [[keras.Input(shape=(1,), name=feature), feature] for feature in features]
        if len(features) == 1:
            all_features = encode_numerical_feature(all_inputs[0][0], all_inputs[0][1], train_ds)
        else:
            all_features = tf.keras.layers.concatenate([encode_numerical_feature(feature_input, feature_name, train_ds) for feature_input, feature_name in all_inputs])

        x = layers.Dense(50, activation="relu")(all_features)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(25, activation="relu")(x)
        output = layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model([fst for fst, _ in all_inputs], output)
        #keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')])
        history = model.fit(train_ds, batch_size=32, epochs=5, validation_data=test_ds)
    print()


def predict_logistic_regression(train_df, test_df, feature_combinations):
    clf = LogisticRegression(random_state=0, multi_class='ovr')

    for features in feature_combinations:
        print(features)
        clf.fit(train_df[features], train_df['label'])
        print(clf.score(test_df[features], test_df['label']))
    print('')

def main(args):
    print('#############################################################################################')
    print(args.train_filepath)
    train_df = pd.read_csv(args.train_filepath, encoding='utf-8')
    train_df.fillna(0, inplace=True)
    train_df = shuffle(train_df, random_state=0)
    test_df = pd.read_csv(args.test_filepath, encoding='utf-8')
    test_df.fillna(0, inplace=True)
    test_df = shuffle(test_df, random_state=0)


    feature_combinations = [
        ['adamic_adar', 'pyramid_match'],
        ['adamic_adar_normalized', 'pyramid_match_normalized'],
        ['adamic_adar_normalized'],

        ['adamic_adar_normalized', 'jaccard_similarity'],
        ['adamic_adar', 'jaccard_similarity'],
        ['jaccard_similarity'],

        ['propagation_normalized'],
        ['propagation', 'adamic_adar_normalized'],
        ['propagation', 'adamic_adar'],
        ['propagation_normalized', 'adamic_adar_normalized'],

        ['adamic_adar', 'weisfeiler_pyramid_match'],
        ['adamic_adar', 'weisfeiler_pyramid_match_normalized'],
        ['adamic_adar_normalized', 'weisfeiler_pyramid_match_normalized']

    ]
    predict_logistic_regression(train_df, test_df, feature_combinations)
    predict_nn(train_df, test_df, feature_combinations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        epilog="Example: python generate_dataset.py"
    )
    parser.add_argument(
        "--train-filepath",
        help="The path for the train dataset",
        dest="train_filepath",
        type=str,
        required=True
    )
    parser.add_argument(
        "--test-filepath",
        help="The path for the test dataset",
        dest="test_filepath",
        type=str,
        required=True
    )
    args = parser.parse_args()

    main(args)
