# Shall I work with them? A ‘knowledge graph’-based approach for predicting future research collaborations
This repository hosts code for the paper: Shall I work with them? A ‘knowledge graph’-based approach for predicting future research collaborations.

## Installation
**Prequisites:**  
* Python 3.6
* Neo4j (version 4.1.8 or greater).  

### Setup
```bash
git clone https://github.com/nkanak/cordkel
cd cordkel
pip install -r requirements.txt
```  
## Datasets
Available [here](https://github.com/nkanak/cordkel/tree/main/data/datasets)

## Experiments
For our experiments we have already generated the datasets found in the link above,
so you only need to run the `predict_all.py` script.

## Data Generation
To generate the data for our experiements, we firstly create the database from the Neo4j desktop app using 4.1.8 as the min. version.  
Then we set the `dbms.memory.heap.max_size = 4G` in the database settings and we install the Graph Data Science plugin (version 1.4.1).
We run the `CORD19_GraphOfDocs.py` script which creates thousands of nodes, and millions of relationships in the database.  
Once it is done, the database is initialized and ready for use. We generate the base datasets, by running `generate_datasets.py`.
Finally, we produce all similarity features for each generated `.csv` by running each `.py` script postfixed with `_all`.  
