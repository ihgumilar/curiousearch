# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: nyampe
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Libraries

# %%
# Libraries
import streamlit as st
import time
from llama_index.llms import MistralAI
from llama_index.embeddings import MistralAIEmbedding
from llama_index import ServiceContext
from llama_index import set_global_service_context
from resources import Toolkit


# OPENAI_API_KEY="xxxx"
MISTRAL_API_KEY="xxx"
METAPHOR_API_KEY="xxx"

# Initialize toolkit
toolkit = Toolkit(metaphor_api_key=METAPHOR_API_KEY)

# %% [markdown]
# # Set up custom LLM and Service Context

# %%
# NOTE: Comment this if you are using OpenAI

# Set up custom LLM and Service Context
llm = MistralAI(api_key=MISTRAL_API_KEY)
embed_model_name = "mistral-embed"
embed_model = MistralAIEmbedding(model_name=embed_model_name, api_key=MISTRAL_API_KEY)

service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
# Set a service context as the global default that applies to the entire LlamaIndex pipeline
set_global_service_context(service_context)


# %% [markdown]
# # Setup

# %%
# Set up OpenAI

# NOTE: Uncomment this if you are using OpenAI
# import openai
# from llama_index.agent import OpenAIAgent
# openai.api_key = OPENAI_API_KEY


# %% [markdown]
# # Streamlit

# %%
st.title("CuriouSearch")
doc_summary_index = None  # Initialize outside the block
showed_new_query_result = False
# st.session_state.messages = [] # Initialize state messages

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("What's your curiosity today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):

        message_placeholder = st.empty()
        full_response = ""
        # Call function to begin a new query
        new_query, doc_summary_index = toolkit.new_query(prompt)

        #Call function to add a new line after printing "Source:Website Link"
        full_response = toolkit.new_line_after_web_link(new_query)
        
        for chunk in new_query.split("\n\n"):
            # Add a new line before printing "Source:Website Link"
            full_response += chunk + " "
            if "?." in full_response:
                full_response = full_response.replace("?. ", "?.\n")
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        # st.session_state.messages.append({"role": "assistant", "content": full_response})



