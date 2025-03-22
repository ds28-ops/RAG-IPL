import streamlit as st
import sqlite3
import re
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from rapidfuzz import process, fuzz

def get_ipl_stats_answer(user_query, db_path="ipl_data.db", openai_api_key=None):
    if openai_api_key is None:
        raise ValueError("OpenAI API key is required")

    # ---------------------------
    # HELPER FUNCTIONS
    # ---------------------------

    def get_all_players(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT player_name FROM batting_stats
            UNION
            SELECT DISTINCT player_name FROM bowling_stats
        """)
        players = [row[0] for row in cursor.fetchall()]
        conn.close()
        return players

    def extract_candidate_names(query):
        # Looks for phrases like "Virat Kohli", "MS Dhoni"
        matches = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', query)
        return matches

    def normalize_player_name_in_query(user_query, db_path, threshold=50):
        db_names = get_all_players(db_path)
        candidate_names = extract_candidate_names(user_query)
        print(f"Candidate names: {candidate_names}")

        for name in candidate_names:
            match, score, _ = process.extractOne(name, db_names, scorer=fuzz.token_sort_ratio)
            print(f"match: {match}, score: {score}")
            if score >= threshold:
                user_query = re.sub(rf'\b{re.escape(name)}\b', match, user_query)

        return user_query

    # ---------------------------
    # CLEAN USER QUERY
    # ---------------------------

    clean_query = normalize_player_name_in_query(user_query, db_path)
    print("Cleaned Query:", clean_query)

    # ---------------------------
    # LOAD SCHEMA & TEAM NAMES
    # ---------------------------

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    table_declarations = [row[0] + ";" for row in cursor.fetchall()]
    table_declarations = [re.sub(r'--.*?($|\n)', r'\1', d) for d in table_declarations]
    all_table_declarations = "\n".join(table_declarations)

    team_query = "SELECT DISTINCT(batting_team) FROM match_ball_by_ball"
    team_names = cursor.execute(team_query).fetchall()
    conn.close()

    # ---------------------------
    # PROMPT TEMPLATE - SQL
    # ---------------------------

    sql_prompt_template = PromptTemplate(
        input_variables=["query_str", "team_names", "all_table_declarations"],
        template="""
You are a master cricket statistician with access to a cricket database and a master at SQL.
You are given tables with data about IPL stats. batting_stats and bowling_stats has season wise batting and bowling stats. match_ball_by_ball has ball by ball information for all the matches. There are other tables as shown in the sql table definitions below. You do not have to use all the tables, use only the required ones.
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
Example:
SELECT player_name, strike_rate 
FROM batting_stats 
WHERE season = 2023 AND balls_faced >= 100 
ORDER BY strike_rate DESC 
LIMIT 5;
Make sure you ONLY output an SQL query and no explanation.
Example:
SELECT 
    striker, 
    SUM(runs_off_bat) AS runs_scored, 
    COUNT(ball_number) AS balls_faced, 
    ROUND((SUM(runs_off_bat) * 100.0) / COUNT(ball_number), 2) AS strike_rate
FROM match_ball_by_ball 
WHERE striker = 'LS Livingstone' 
  AND ball_number BETWEEN 15.1 AND 20.6 
GROUP BY striker;
"""
    )

    # ---------------------------
    # GENERATE SQL QUERY
    # ---------------------------

    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4", temperature=0)
    sql_chain = LLMChain(llm=llm, prompt=sql_prompt_template)
    sql_chain_output = sql_chain.invoke({
        "query_str": clean_query,
        "team_names": team_names,
        "all_table_declarations": all_table_declarations
    })
    sql_query = sql_chain_output.get("text", "").strip() if isinstance(sql_chain_output, dict) else sql_chain_output.strip()
    print("Generated SQL Query:", sql_query)

    # ---------------------------
    # EXECUTE SQL
    # ---------------------------

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        sql_result = cursor.execute(sql_query).fetchall()
    except Exception as e:
        conn.close()
        return f"SQL Error: {e}"
    conn.close()

    # ---------------------------
    # GENERATE FINAL ANSWER
    # ---------------------------

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

    answer_chain = LLMChain(llm=llm, prompt=answer_prompt_template)
    answer_chain_output = answer_chain.invoke({
        "query_str": user_query,  # original uncleaned user query
        "json_table": json.dumps(sql_result),
        "sql_query": sql_query
    })
    final_answer = answer_chain_output.get("text", "").strip() if isinstance(answer_chain_output, dict) else answer_chain_output.strip()

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
