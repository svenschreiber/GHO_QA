import pandas as pd
import sqlite3
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
import ollama
import re
from tqdm import tqdm

def pull_ollama_model(model, verbose=False):
    if not verbose: 
        ollama.pull(model)
        return

    current_digest, bars = '', {}
    for progress in ollama.pull(model, stream=True):
        digest = progress.get('digest', '')
        if digest != current_digest and current_digest in bars:
            bars[current_digest].close()
        if not digest:
            print(progress.get('status'))
            continue
        if digest not in bars and (total := progress.get('total')):
            bars[digest] = tqdm(total=total, desc=f'pulling {digest[7:19]}', unit='B', unit_scale=True)
        if completed := progress.get('completed'):
            bars[digest].update(completed - bars[digest].n)
        current_digest = digest

def extract_sql(response, intermediate=False):
    rules = [r"\bWITH\b .*?;", r"SELECT.*?;", r"```sql\n(.*)```", r"```(.*)```"]
    for rule in rules:
        if sqls := re.findall(rule, response, re.DOTALL): 
            return sqls[0] if intermediate else sqls[-1]
    return response

class SQLRAG:
    def __init__(self, ollama_model="phi4", db_path="data/gho.db", verbose_initalization=False):
        self.ollama_model = ollama_model
        self.conn = sqlite3.connect(db_path)

        embedding_func = DefaultEmbeddingFunction()
        chroma_client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
        self.table_collection = chroma_client.get_or_create_collection(name="tables", embedding_function=embedding_func)
        self.store_tables_ddls()

        pull_ollama_model(ollama_model, verbose_initalization)
    
    def store_tables_ddls(self):
        ddls = pd.read_sql_query("SELECT type, sql FROM sqlite_master WHERE sql is not null", self.conn)
        ddls = ddls['sql'].to_list()
        self.table_collection.add(documents=ddls, ids=[f"id{i}" for i in range(len(ddls))])
    
    def get_system_prompt(self, user_prompt):
        system_prompt = (
            "You are an SQLite query expert. Generate an optimized and accurate query based solely on the user's question, ensuring it follows the given context, response guidelines, and format instructions. \n\n"
        )

        system_prompt += "===Tables \n"
        ddls = self.table_collection.query(query_texts=user_prompt, n_results=min(self.table_collection.count(), 10))["documents"][0]
        for ddl in ddls:
            system_prompt += ddl + "\n\n"
            table_name = re.search(r'CREATE TABLE\s+"([^"]+)"', ddl, re.IGNORECASE).group(1)
            df = pd.read_sql_query(f"SELECT * FROM \"{table_name}\" WHERE rowid IN ( SELECT rowid FROM ( SELECT rowid, value FROM \"{table_name}\" GROUP BY value ORDER BY RANDOM() LIMIT 5));", self.conn)
            system_prompt += f"Five random table rows:\n{df.to_markdown()}\n\n"

        system_prompt += (
            "===Response Guidelines \n"
            "1. Generate valid SQL if the context is sufficient. Ensure it is SQLite-compliant and error-free. \n"
            "2. Use the most relevant tables only. \n"
            "3. Use the example rows provided with each table, to ensure the right values are selected. \n"
            "4. Use LIKE for filtering TEXT columns unless otherwise specified. \n"
            "5. Avoid unnecessary complexity. Make the queries as short as possible. Do not focus on too many things at once. \n"
        )
        return system_prompt
    
    def get_sql(self, prompt, verbose=False):
        response = ollama.chat(model=self.ollama_model, messages=prompt, options=dict(num_ctx=16384))["message"]["content"]
        if verbose: print(response)
        sql = extract_sql(response).lower()
        return sql
    
    def prompt(self, question, verbose=False, print_results=False):
        messages = [
            {'role': 'system', 'content': self.get_system_prompt(question)},
            {'role': 'user', 'content': question},
        ]

        if verbose:
            print("The sytem prompt is:")
            print(messages[0]["content"])

        sql = self.get_sql(messages)
        if verbose: print(sql)
        try:
            df = pd.read_sql_query(sql, self.conn)
        except Exception as e:
            print("Sorry, but there was an error generating a SQL query for your question. Please consider formulating it differently.")
            if verbose: print(repr(e))
            return

        if print_results: 
            print(df.to_markdown())
            print()

        system_prompt = (
            f"You are a helpful data scientist. Question: '{question}'\n\n"
            f"SQL query used:\n{sql}\n\n"
            f"Here is the query result as a pandas DataFrame:\n\n{df.to_markdown()}\n\n"
        )

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': "Summarize the data concisely based on the question. Use the mentioned SQL query to contextualize the data. Don't overcomplicate your answer, highlight only the important information of the data. DON'T MENTION the SQL query and in which format the data was given to you. Summarize directly the data."},
        ]

        response = ollama.chat(model=self.ollama_model, messages=messages, options=dict(num_ctx=16384), stream=True)
        for chunk in response:
            print(chunk["message"]["content"], end="", flush=True)
        print()

if __name__ == "__main__":
    model = SQLRAG()
    print("What is your question?")
    question = input(">>> ")
    model.prompt(question, print_results=True, verbose=True)
    #model.prompt("What is the number of HIV infections in germany for each year and does it have a correlation with diabetes rates?", print_results=True)