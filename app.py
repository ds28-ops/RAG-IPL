import streamlit as st
import sqlite3
import re
import json
from langchain_openai import ChatOpenAI  # Make sure you installed langchain-openai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def get_ipl_stats_answer(user_query, db_path="ipl_data.db", openai_api_key=None):
    if openai_api_key is None:
        raise ValueError("OpenAI API key is required")
        
    # Connect to the database and extract all CREATE TABLE statements
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    table_declarations = [row[0] + ";" for row in cursor.fetchall()]

    # Remove SQL comments from declarations
    def remove_sql_comments(declarations):
        return [re.sub(r'--.*?($|\n)', r'\1', d) for d in declarations]
    table_declarations = remove_sql_comments(table_declarations)
    all_table_declarations = "\n".join(table_declarations)
    
    # Retrieve distinct team names from the matches table
    team_query = "SELECT DISTINCT(batting_team) FROM matches"
    team_names = cursor.execute(team_query).fetchall()
    conn.close()

    # Define the prompt template for generating the SQL query
    sql_prompt_template = PromptTemplate(
        input_variables=["query_str", "team_names", "all_table_declarations"],
        template="""
            You are a master cricket statistician with access to a cricket database and a master at SQL.
            You are given tables with data about IPL stats. There is a separate table for batting and bowling stats for each year like batting_stats_2012 and bowling_stats_2012 and so on for every year. There is also a matches table which has ball by ball data for every match played. historical_batting_stats and historical_bowling_stats have the overall stats including all the seasons. You do not have to use all the tables, use only the required ones.
            Here are the formulas for basic cricket statistics that you should use if relevant to the user's query:
            
            Batting Average = (Total Runs Scored) / (Number of Times Dismissed)  
            Batting Strike Rate = (Total Runs Scored / Total Balls Faced) * 100  
             Bowling Economy Rate = (Total Runs Conceded) / ((Balls Bowled / 6))  
            Bowling Strike Rate = (Total Balls Bowled) / (Total Wickets Taken) 
            The team names are:
            {team_names}
            
            Generate a SQL query to answer the following question from the user:
            "{query_str}"
            
            The SQL query should use only tables with the following SQL definitions:
            
            {all_table_declarations}
            Output only the raw SQL query string with NO ADDITIONAL FORMATTING and MARKDOWN.
            Example-
            SELECT total_runs FROM historical_batting_stats WHERE player_name = 'MS Dhoni';
            Make sure you ONLY output an SQL query and no explanation.
            """
                )

    # Create a ChatOpenAI instance
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4", temperature=0)
    
    # Build the SQL generation chain and invoke it
    sql_chain = LLMChain(llm=llm, prompt=sql_prompt_template)
    sql_chain_output = sql_chain.invoke({
        "query_str": user_query,
        "team_names": team_names,
        "all_table_declarations": all_table_declarations
    })
    if isinstance(sql_chain_output, dict):
        sql_query = sql_chain_output.get("text", "").strip()
    else:
        sql_query = sql_chain_output.strip()
    
    # Execute the generated SQL query on the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    sql_result = cursor.execute(sql_query).fetchall()
    conn.close()
    
    # Define the prompt template for generating the final natural language answer
    answer_prompt_template = PromptTemplate(
        input_variables=["query_str", "json_table", "sql_query"],
        template="""
You are a cricket stats guru.
You are answering questions about Indian Premier League stats. Use the following to answer to the best of your ability.
Use natural language in your answer – you can be funny, witty, or sarcastic if required.

USER QUERY: {query_str}

JSON table:
{json_table}

This table was generated by the following SQL query:
{sql_query}

Answer ONLY using the information in the table and the SQL query, and if the
table does not provide the information to answer the question, answer
"No Information".
"""
    )
    
    # Build the final answer chain and invoke it
    answer_chain = LLMChain(llm=llm, prompt=answer_prompt_template)
    answer_chain_output = answer_chain.invoke({
        "query_str": user_query,
        "json_table": json.dumps(sql_result),
        "sql_query": sql_query
    })
    if isinstance(answer_chain_output, dict):
        final_answer = answer_chain_output.get("text", "").strip()
    else:
        final_answer = answer_chain_output.strip()
    
    return final_answer

# Streamlit App Interface
st.title("IPL Stats Chatbot")
st.write("Ask your IPL query and get statistics!")
st.write("Created by [Dhruv Sridhar](https://www.linkedin.com/in/dhruv-sr/)")
st.write("Example queries: Who are the top 5 run-scorers in IPL 2021?, Who are the top 10 bowlers with lowest economy rate in IPL history with a minimum of 1000 balls bowled?")
st.write("Note: This demo app might not work for all queries. Currently supports only a subset of IPL queries.")

user_query = st.text_area("Enter your IPL query:", height=150)

# Retrieve OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["openai"]["api_key"]

if st.button("Get Answer"):
    if user_query:
        with st.spinner("Processing..."):
            answer = get_ipl_stats_answer(user_query, openai_api_key=openai_api_key)
        st.subheader("Answer:")
        st.write(answer)
    else:
        st.error("Please enter a query.")
