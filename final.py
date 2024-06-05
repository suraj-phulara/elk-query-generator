import os
import json
import streamlit as st
from openai import OpenAI
from opensearchpy import OpenSearch
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from time import sleep
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from querymodel import ProductQuerySchema

load_dotenv()
structured_index_name = os.getenv("structured_index_name")


def chat_gpt(query, model="gpt-3.5-turbo-0301"):

    model = ChatOpenAI(temperature=0)

    # And a query intented to prompt a language model to populate the data structure.
    # query = f"read this text extracted from a user resume carefully and classify this based on different criteria into the json format.   text : {text} \n\n"

    # Set up a parser + inject instructions into the prompt template.
    parser = JsonOutputParser(pydantic_object=ProductQuerySchema)

    prompt = PromptTemplate(
        template="understand the entire problem carefully and anwer it \n{query}\n only give me the json back nothing else",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | model | parser

    response = chain.invoke({"query": query})
    

    return response

def chat_gpt2(query, model="gpt-3.5-turbo-0301"):

    model = ChatOpenAI(temperature=0)

    # And a query intented to prompt a language model to populate the data structure.
    # query = f"read this text extracted from a user resume carefully and classify this based on different criteria into the json format.   text : {text} \n\n"

    # Set up a parser + inject instructions into the prompt template.
    parser = JsonOutputParser()

    prompt = PromptTemplate(
        template="understand the entire problem carefully and anwer it \n{query}\n only give me the json back nothing else",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | model | parser

    response = chain.invoke({"query": query})
    

    return response


def es_connect():
    try:
        es = OpenSearch(
            hosts=[{'host': 'localhost', 'port': 9200}],
            http_auth=('admin', 'admin@123'),
            use_ssl=False,
            verify_certs=False,  # Set to True if you have proper SSL certificates
            ssl_show_warn=False
        )
        # Optional: Check if the connection is successful by making a simple request
        es.info()
        print("Connection established")
        return es

    except Exception as e:
        st.error(f"Failed to connect to OpenSearch cluster: {e}")
        return None



es = es_connect()



def execute_query(index_name, query_dsl):
    try:
        # Execute the search query
        response = es.search(index=index_name, body=query_dsl)
        return response['hits']['hits']
    except Exception as e:
        # st.error(f"Failed to execute OpenSearch query: {e}")
        return []

def main():
    try:
        with open('abc.json', 'r', encoding='utf-8') as json_file:
            open_search_data = json.load(json_file)
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse JSON file: {e}")
        return

    st.set_page_config(
        "ElastiGenius",
        page_icon="https://media.licdn.com/dms/image/C4E0BAQErzHJr7lA-uQ/company-logo_200_200/0/1631356294168?e=1714608000&v=beta&t=lbeplkUBiGsPGvObCIUmLk5qRA9X8NvoJGHWBZEC6so",
        layout="wide"
    )
    st.title("ElastiGenius: AI-Powered Product Discovery")

    query = st.text_input("Enter your query:", key="user_query", value=st.session_state.get('last_query', ''))
    button = st.button("Generate Query")

    if button and query:

        prompt2 = f"""

        carefully analyze the human language query :{query} given and under every aspect of it and classify the data between fields like name , main_subcategory, price etc..
        
        note: leave all the fields not presen in the human query

        Below JSON data contains categories and their corresponding subcategories. The top-level keys represent main categories, and each main category maps to a list of its subcategories:
        categories_info: {json.dumps(open_search_data.get('categories', {}), indent=4)}


        now classify what the user is looking for based on this :

        
        "actual_price": '',
        "discount_price": '',
        "main_category": '',
        "name": '',
        "no_of_ratings": '',
        "ratings": '',
        "sub_category": ''

        LEAVE ANY FIELD THAT THE USER DID NOT WANT TO Secified but you must include everything the user specified for example the price range , etc ((most important))


        note that under no circumstances are you allowed to add ny data by yourself only use what the user provided you

        also treat actual price and discounted price as alias to each other

        note: give extra care to if the user gave a price range or not . never miss this critical info
        """


        answer = chat_gpt(prompt2)
        st.write(answer)

        prompt = f"""

            The query user have in json format -   Query : {answer}

            Carefully analyze and understand the below prompt and give the solution:

            Below is the mapping of some index in my OpenSearch database:
            mappings: {json.dumps(open_search_data.get('mappings', {}), indent=4)}


            now based on the query 


            Now your task is to properly understand the structure, mapping, and user queries as best as you can and then generate an OpenSearch query using that information so that I can execute that query in OpenSearch database through Kibana and return the output to the users.


            [Note]
            - Only print the Query, nothing else. No explanation and opening text etc.
            - Incorporate keywords like "most popular," "famous," or "top" into the search query to prioritize highly-rated items.
            - Also ensure to strictly follow the OpenSearch Mapping.
            - refrain from employing nested queries important.
            - Also under no circumstances are you allowed to make up any data by yourself in the query only and only use the data i provided you strictly
            - Do not add any fields like Range or sorting, number of ratings, actual price, discounted price, or any other integer fields that are not provided in the user query.
            - Final and most important do not include any made up data by yourself only take the data from this prompt and user query and leave the other fields for which data is not present in the user query
            - range if exists it should always be inside should clause
            - always remember the only thing that comes under must field is only and only name nothing else 
            [Schema]
            - Categories fields should be inside (Should clause) as sometimes in our data the item can exist in wrong categories also.
            - Also for any Range or sorting, number of ratings, actual price, discounted price, or any other integer fields should also be inside (Should clause) only and  only if provided in the user query..
            - Name should be inside must/match clause.


            note that under no circumstances are you allowed to add ny data by yourself only use what the I provided you in the json only
            also treat actual price and discounted price as alias to each other

            now Generate the output below:


            also remember the name field should always has most importance and weitage in the searching for the matching results so adjust the query accordingly

            
        """

        

        # st.write(prompt)
        

        answer = chat_gpt2(prompt)
        st.write(answer)
        
        query_dsl = answer
        results = execute_query(structured_index_name, query_dsl)
        st.write(results)


if __name__ == "__main__":
    main()

