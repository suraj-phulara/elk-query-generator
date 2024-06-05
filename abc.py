
import os

import pandas as pd
import json
import streamlit as st
from openai import OpenAI
from opensearchpy import OpenSearch
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.pydantic_v1 import BaseModel, Field
from time import sleep
from dotenv import load_dotenv


load_dotenv()

# os.environ['GOOGLE_API_KEY'] = "AIzaSyCCKXZFrT43tKnrePgpfcrgF5svcgpUflQ"
# Assuming OPENAI_API_KEY is set as an environment variable.

if 'retries' not in st.session_state:
    st.session_state.retries = 0

# Define max retries
MAX_RETRIES = 3

class main_category(BaseModel):
    main_categry: str = Field(description="Main category of the product to be retrived") 

def es_connect(user):
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


def execute_query(index_name, query_dsl):
    try:
        # Execute the search query
        response = es.search(index=index_name, body=query_dsl)
        return response['hits']['hits']
    except Exception as e:
        # st.error(f"Failed to execute OpenSearch query: {e}")
        return []


def display_results(hits, query):
    data = []
    for hit in hits:
        name = hit['_source'].get('name','N/A')
        sub_category = hit['_source'].get('sub_category', "N/A")
        main_category = hit['_source'].get('main_category',"N/A")
        no_of_ratings = hit['_source'].get('no_of_ratings', 'N/A')
        ratings = hit['_source'].get('ratings', 'N/A')
        actual_price = hit['_source'].get('actual_price', 'N/A')
        discount_price = hit['_source'].get('discount_price', 'N/A')
        link = hit['_source'].get('link', 'N/A')

        data.append({
            'Name': name,
            'Sub Category': sub_category,
            'Main Category': main_category,
            'No. of Ratings': no_of_ratings,
            'Ratings': ratings,
            'Actual Price': actual_price,
            'Discount Price': discount_price,
            'Link': link
        })

    prompt = f"""Imagine you are a customer engagement expert with a knack for turning product features into compelling benefits.
      Given a user's query and comprehensive products information, your task is to create a response that not only answers the query but also positions the product as the ideal solution to the user's needs. I user wants comparisoin than give him an comprehsive answer compareing all results.
      strictly follow Print all products in Tabular form only comparing all the important factors for making the output more appealing to users.
      Incorporate key product details to construct a narrative that resonates with the user's situation, emphasizing how the product stands out from competitors. 
      Your response should be informative, engaging, and persuasive, aimed at an audience considering a purchase in the product's category.
      
      user's query :- {query}
      comprehensive products information :- {data}
      """
    answer = chat_gpt(prompt)
    progress_bar.progress(0.4,"Fetching Relevant articles")
    sleep(0.5)
    progress_bar.progress(0.5,"Analysing Articles")
    sleep(0.5)
    progress_bar.progress(0.6)
    sleep(0.5)
    progress_bar.progress(0.8,"Crafting responnse tailored to your need")
    sleep(0.3)
    progress_bar.progress(0.9)
    sleep(0.5)
    progress_bar.progress(1.0)
    progress_bar.empty()
    st.write(answer)
    return answer

def fetch_mapping(es, index_name):
    mapping = es.indices.get_mapping(index=index_name)
    st.toast("Mapping fetched Successfully")
    return mapping

def fetch_categories(es, index_name, field_name):
    agg_query = {
        "size": 0,
        "aggs": {
            "unique_values": {
                "terms": {"field": field_name}  # Adjust size as needed
            }
        }
    }
    response = es.search(index=index_name, body=agg_query)
    categories = [bucket['key'] for bucket in response['aggregations']['unique_values']['buckets']]
    st.toast("Categories fetched Successfully")
    return categories

def fetch_subcategories(es, index_name, main_category):
    agg_query = {
        "size": 0,
        "query": {
            "match": {"main_category": main_category}
        },
        "aggs": {
            "unique_values": {
                "terms": {"field": "sub_category"}  # Adjust size as needed
            }
        }
    }
    response = es.search(index=index_name, body=agg_query)
    subcategories = [bucket['key'] for bucket in response['aggregations']['unique_values']['buckets']]
    st.toast("Subcategories fetched Successfully")
    return subcategories

def chat_gpt(prompt, model="gpt-3.5-turbo-0301"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(model=model,
                                              messages=[{"role": "system", "content": "You are an OpenSearch assistant."}, {"role": "user", "content": prompt}])
    return response.choices[0].message.content

st.set_page_config("ElastiGenius", page_icon="https://media.licdn.com/dms/image/C4E0BAQErzHJr7lA-uQ/company-logo_200_200/0/1631356294168?e=1714608000&v=beta&t=lbeplkUBiGsPGvObCIUmLk5qRA9X8NvoJGHWBZEC6so",layout="wide")
st.title("ElastiGenius: AI-Powered Product Discovery")

elastic_user = os.getenv("elastic_user")
structured_index_name = os.getenv("structured_index_name")

es = es_connect(elastic_user)
agg_query = {
    "size": 0,
    "aggs": {
        "unique_values": {
            "terms": {"field": "main_category"}  # Adjust size as needed
        }
    }
}
response = es.search(index=structured_index_name, body=agg_query)
print("response--->",response)
st.toast("Connected to Elastic Search")

if es:
    mapping = fetch_mapping(es, structured_index_name)
    main_categories = fetch_categories(es, structured_index_name, "main_category")
    # st.write(main_categories)
    query = st.text_input("Enter your query:", key="user_query", value=st.session_state.get('last_query', ''))
    main_category_template = f"""Fetch the main category that is most suitable to the user query.
            main categories - {main_categories}
            user query :- {query}

            -Main categories should be from the provided list only.
            -Only print the category name nothing else no explanation and opening text etc.
            """

    parser = PydanticOutputParser(pydantic_object=main_category)

    main_category_prompt = PromptTemplate(
            template=main_category_template,
            input_variables=["main_categories", "query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
                )
    
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    main_chain = LLMChain(llm=llm, prompt=main_category_prompt, verbose= True,output_key='main_category')

    main = main_chain.invoke({"query": query, "main_categories": main_categories})

    cate = main['main_category']

    st.write(main['main_category'])

    sub_categories = fetch_subcategories(es, structured_index_name, main['main_category'])

    st.write(sub_categories)
    button = st.button("Generate Query")

    # Add the retry mechanism after the ChatGPT response is received

    if button and query:
        retry_attempts = 0
        max_retries = 3
        results_fetched = False

        progress_bar = st.progress(0)

        while not results_fetched and retry_attempts < max_retries:

            progress_value = 0
            progress_bar.progress(progress_value)

            prompt = f"""Task: Generate OpenSearch Query

                        [Input]
                        - OpenSearch Mapping: {mapping}
                        - User Query: {query} 
                        - Main Category: {cate}
                        - Sub Categories: {sub_categories}

                        [Note]
                        - Ensure selecting best sub category exclusively from provided lists. 
                        - Only print the Query nothing else no explanation and opening text etc. 
                        - Please refrain from employing nested queries.
                        - Kindly ensure that your query should align with the Main Categories and Sub Categories listed above. Also don't use more than 2 categories in the generated query.
                        - Kindly abstain from modifying types categorized as 'keyword' in the respective type, maintain the original format of entries. This guideline extends to all type classified as 'keyword' in mapping.
                        - Also ensure to strictly follow the OpenSearch Mapping.
                        - Incorporate keywords like "most popular," "famous," or "top" into the search query to prioritize highly-rated items.
                        [Schema]
                        - Categories should be filtered inside (filter clause)
                        - for any Range or sorting, number of rating, actual price, discounted price, or any other integer fields should be inside (Should clause)
                        - Name should be inside must/match clause.
                        - Use term query not terms query.
                        """

            
            answer = chat_gpt(prompt)                
            try:
                query_dsl = json.loads(answer)
                results = execute_query(structured_index_name, query_dsl)
                if results:
                    progress_bar.progress(0.2,"Fetching results")
                    sleep(0.3)
                    st.code(answer)
                    display_results(results, query)
                    results_fetched = True
                else:
                    # st.write("No results found, trying again...")
                    continue
            except json.JSONDecodeError:
                # st.write("Failed to parse the ChatGPT-generated query, retrying...")
                pass
            
            retry_attempts += 1
            print(retry_attempts)
            print(MAX_RETRIES - retry_attempts, "Attempts left.")

        if not results_fetched:
            progress_bar.progress(1.0)
            st.error("Failed to generate and execute a valid query after several attempts. Please try again or modify your query.")
