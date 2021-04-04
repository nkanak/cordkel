import utils
from CORD19_GraphOfDocs.neo4j_wrapper import Neo4jDatabase
from grakel.kernels import PyramidMatch
from grakel import Graph
from CORD19_GraphOfDocs.utils import read_file
from CORD19_GraphOfDocs.utils import generate_words
import json

import argparse


def create_text(file_text, fieldname='abstract'):
    obj = json.loads(file_text)
    text = ''.join(item['text'] for item in obj[fieldname])
    return text


def create_author_graph_of_words(docs, window_size=4):
    edges = {}
    unique_words = set()

    for doc in docs:
        for i in range(len(doc)):
            unique_words.add(doc[i])
            for j in range(i + 1, i + window_size):
                if j < len(doc):
                    unique_words.add(doc[j])
                    edge_tuple1 = (doc[i], doc[j])
                    edge_tuple2 = (doc[j], doc[i])
                    if edge_tuple1 in edges:
                        edges[edge_tuple1] += 1
                    elif edge_tuple2 in edges:
                        edges[edge_tuple2] += 1
                    else:
                        edges[edge_tuple1] = 1
    node_labels = {word: word for word in unique_words}
    g = Graph(edges, node_labels=node_labels)

    return g


def get_author_filenames(database, author_name):
    query = (
        'MATCH (a:Author)-[:writes]->(p:Paper) '
        f'WHERE a.name="{author_name}" RETURN a.name, collect(p.filename)'
    )
    return database.execute(query, 'r')


def get_graph_kernel(database, author_name):
    filenames = get_author_filenames(database, author_name)
    docs = [read_file('data/CORD-19-research-challenge/dataset', fname + '.json') for fname in filenames[0][1]]
    docs = [generate_words(create_text(doc)) for doc in docs]

    g = create_author_graph_of_words(docs, 4)
    # print(len(g.vertices))
    if len(g.vertices) == 0:
        gk = None
    else:
        gk = PyramidMatch(normalize=False)
        gk.fit([g])
    return gk


def get_graph(database, author_name):
    filenames = get_author_filenames(database, author_name)
    docs = [read_file('data/CORD-19-research-challenge/dataset', fname + '.json') for fname in filenames[0][1]]
    docs = [generate_words(create_text(doc)) for doc in docs]

    g = create_author_graph_of_words(docs, 4)
    # print(len(g.vertices))
    if len(g.vertices) == 0:
        g = None
    return g


def calculate_similarity_feature(author1_graph_kernel, author2_graph):
    gk = author1_graph_kernel
    g = author2_graph
    if gk is None or g is None:
        similarity = 0
    else:
        similarity = gk.transform([g])[0][0]
    return similarity


def main(args):
    database = Neo4jDatabase('bolt://localhost:7687', 'neo4j', '1234')
    column_name = args.column_name
    column_name_normalized = f'{column_name}_normalized'

    train_dataset = utils.read_from_csv_file(args.train_filepath)
    test_dataset = utils.read_from_csv_file(args.test_filepath)

    print('train###')
    for i, sample in enumerate(train_dataset):
        author1 = sample['author1']
        author2 = sample['author2']
        print(i, author1, author2)
        author1_graph_kernel = get_graph_kernel(database, author1)
        author2_graph = get_graph(database, author2)
        sample[column_name] = calculate_similarity_feature(author1_graph_kernel, author2_graph)
        train_dataset[i] = sample

    print('test###')
    for i, sample in enumerate(test_dataset):
        author1 = sample['author1']
        author2 = sample['author2']
        print(i, author1, author2)
        author1_graph_kernel = get_graph_kernel(database, author1)
        author2_graph = get_graph(database, author2)
        sample[column_name] = calculate_similarity_feature(author1_graph_kernel, author2_graph)
        test_dataset[i] = sample

    utils.write_list_of_dicts_to_csv_file(args.train_filepath, train_dataset)
    utils.write_list_of_dicts_to_csv_file(args.test_filepath, test_dataset)

    from sklearn.preprocessing import MinMaxScaler
    train_dataset = utils.read_from_csv_file(args.train_filepath)
    test_dataset = utils.read_from_csv_file(args.test_filepath)

    train_values = []
    for sample in train_dataset:
        train_values.append([sample[column_name]])

    test_values = []
    for sample in test_dataset:
        test_values.append([sample[column_name]])

    scaler = MinMaxScaler()
    scaler.fit(train_values + test_values)

    normalized_train_values = scaler.transform(train_values)
    for i, sample in enumerate(train_dataset):
        sample[column_name_normalized] = normalized_train_values[i, 0]
        train_dataset[i] = sample

    normalized_test_values = scaler.transform(test_values)
    for i, sample in enumerate(test_dataset):
        sample[column_name_normalized] = normalized_test_values[i, 0]
        test_dataset[i] = sample

    utils.write_list_of_dicts_to_csv_file(args.train_filepath, train_dataset)
    utils.write_list_of_dicts_to_csv_file(args.test_filepath, test_dataset)


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
    parser.add_argument(
        "--column-name",
        help="The column name to add the calculation",
        dest="column_name",
        type=str,
        required=True
    )
    args = parser.parse_args()

    main(args)