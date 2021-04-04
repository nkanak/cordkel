import csv
import pickle

def read_from_csv_file(filename):
    rows = []
    with open(filename, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def write_to_csv_file(filename, fieldnames, data):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        writer.writerows(data)

def write_list_of_dicts_to_csv_file(filename, data):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, data[0].keys())
        writer.writeheader()
        writer.writerows(data)

def write_object_to_pickle_file(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def read_pickle_file(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj
