# Shall I work with them? A ‘knowledge graph’-based approach for predicting future research collaborations
This repository hosts code for the paper: Shall I work with them? A ‘knowledge graph’-based approach for predicting future research collaborations

## Datasets
Available [here](https://github.com/nkanak/cordkel/tree/main/data/datasets)
# CORD19 Knowledge Graph Representation + Graph-of-docs

## Test Results
Check **predict.py** file.

## Installation
**Prequisites:**  
* Python 3  
* Neo4j (version 3.5.14 or greater).  


### Set up
```bash
git clone https://github.com/NC0DER/CORD19_GraphOfDocs
cd GraphOfDocs
pip install -r requirements.txt
```  

## Database Setup
Create a new database from the `Neo4j` desktop app using 3.5.14 as the min. version.  
Update your memory settings to match the following values,  
and install the following extra plugins as depicted in the image.
![image2](https://github.com/NC0DER/CORD19_GraphOfDocs/blob/master/CORD19_GraphOfDocs/images/settings.jpg)
*Hint: if you use a dedicated server that only runs `Neo4j`, you could increase these values, 
accordingly as specified in the comments of these parameters.*

Run the `CORD19_GraphOfDocs.py` script which will create thousands of nodes, 
and millions of relationships in the database.  
Once it's done, the database is initialized and ready for use. 

## Running the app
You could use the `Neo4j Browser` to run your queries.

## Citation
You can find the bibtex in [this link](), should you want to cite this paper.