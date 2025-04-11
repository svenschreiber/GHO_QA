# NLP Project: Global Health Observatory QA System

This repository contains the code for our SQL and RAG-based question answering system. This project was part of the 'Natural Language Processing and the Web' exercise 2024/25 at the University of Hamburg. Our system can be used to accurately answer questions regarding the [Global Health Observatory (GHO)](https://www.who.int/data/gho) dataset by the World Health Organization. Questions can be formulated in natural language, making it easy to explore and navigate the large amount of data provided in the dataset. 

## System Overview
![image](media/system_overview.svg)

The architecture uses Chroma as a vector store to match questions to related SQL tables. Using Ollama we prompt an LLM with the matching tables and some example data rows to generate a fitting SQL query based on the user-provided question. The SQL query is executed on our SQLite database. Finally, the LLM is prompted again to summarize the SQL results and generate the system's response to the user.

## Usage
Dependencies can be installed via:
```
pip install -r requirements.txt
```
To run the QA system you have to first download and filter the GHO indicators. This can be done with the [`download_indicators`](download_indicators.ipynb) notebook. Note that in this example we only used the _diabetes prevalence_ and _HIV cases_ indicators. To run the model and ask questions, refer to the [`run_model`](run_model.ipynb) notebook.

## Authors
- Sven Schreiber
- Sebastian Winter