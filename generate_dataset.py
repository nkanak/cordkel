import utils
from CORD19_GraphOfDocs.neo4j_wrapper import Neo4jDatabase
from CORD19_GraphOfDocs.select import create_graph_features
from CORD19_GraphOfDocs import select
from sklearn.preprocessing import MinMaxScaler
import uuid

import argparse


def main(args):
    database = Neo4jDatabase('bolt://localhost:7687', 'neo4j', '1234')
    fieldnames = ['author1', 'author2', 'adamic_adar', 'common_neighbors', 'preferential_attachment', 'total_neighbors', 'jaccard_similarity', 'label', 'adamic_adar_normalized']

    train_positive = select.get_positive_examples(database, args.number_of_train_positives, train_set=True)
    train_negative = select.get_negative_examples(database, args.number_of_train_negatives, train_set=True)
    train_dataset = train_positive + train_negative

    train_samples = create_graph_features(database, train_dataset, True)

    test_positive = select.get_positive_examples(database, args.number_of_test_positives, train_set=False)
    test_negative = select.get_negative_examples(database, args.number_of_test_negatives, train_set=False)
    test_dataset = test_positive + test_negative

    test_samples = create_graph_features(database, test_dataset, False)

    scaler = MinMaxScaler()
    scaler.fit([[sample[2]] for sample in train_samples] + [[sample[2]] for sample in test_samples])

    for i, sample in enumerate(train_samples):
        train_samples[i].append(scaler.transform([[sample[2]]])[0][0])
    for i, sample in enumerate(test_samples):
        test_samples[i].append(scaler.transform([[sample[2]]])[0][0])

    random_postfix = uuid.uuid4().hex
    utils.write_to_csv_file(f'{args.output_dir}/train_dataset_{random_postfix}.csv', fieldnames, train_samples)
    utils.write_to_csv_file(f'{args.output_dir}/test_dataset_{random_postfix}.csv', fieldnames, test_samples)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        epilog="Example: python generate_dataset.py"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory containing the generated dataset",
        dest="output_dir",
        type=str,
        required=True
    )
    parser.add_argument(
        "--number-of-train-positives",
        help="Number of positive samples for the train dataset",
        dest="number_of_train_positives",
        type=int,
        required=True
    )
    parser.add_argument(
        "--number-of-train-negatives",
        help="Number of negative samples for the train dataset",
        dest="number_of_train_negatives",
        type=int,
        required=True
    )
    parser.add_argument(
        "--number-of-test-positives",
        help="Number of positive samples for the test dataset",
        dest="number_of_test_positives",
        type=int,
        required=True
    )
    parser.add_argument(
        "--number-of-test-negatives",
        help="Number of negative samples for the test dataset",
        dest="number_of_test_negatives",
        type=int,
        required=True
    )
    args = parser.parse_args()

    main(args)