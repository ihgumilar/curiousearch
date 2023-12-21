from llama_index import get_response_synthesizer
from llama_index.indices.document_summary import DocumentSummaryIndex
import re
from llama_hub.tools.metaphor.base import MetaphorToolSpec
import time
import streamlit as st

class Toolkit:
    """
    A class that provides various tools for processing user queries and generating summaries.
    """
    
    def __init__(self, metaphor_api_key):
        """
        Initializes the Toolkit with the provided Metaphor API key.

        Args:
            metaphor_api_key (str): The API key for the Metaphor tool.
        """
        # Set up Metaphor tool
        self.metaphor_tool = MetaphorToolSpec(api_key=metaphor_api_key)

    
    # @st.cache_data
    def new_query(self,user_message):
        """
        Searches the internet for a user message, retrieves web content, and creates a summary.

        Args:
            user_message (str): The user's query.

        Returns:
            Tuple[str, DocumentSummaryIndex]: A tuple containing concatenated summaries and the summary index.
        """      
        with st.status("Serving your request...", expanded=True) as status:

            # Search in internet
            search_results = self.metaphor_tool.search(user_message, num_results=3)

            # Populate extracted results into documents list & Insert link as metadata
            documents = []

            # Showing relevant websites
            st.write("Searching for data...")
            time.sleep(4)
            
            # Showing relevant websites
            st.write("Found websites...")
            time.sleep(1)
            
            for idx, search_result in enumerate(search_results):
                # Extract url
                url = search_results[idx]['url']
                # Extract id
                id = search_results[idx]['id']
                # Retrieve web content
                content = self.metaphor_tool.retrieve_documents(id)
                # Populate all contents
                documents.extend(content)
                # Adding index doc_id so that it will be easy to use instead of random number
                documents[idx].doc_id = str(idx)  # need to be string, otherwise the metadata, e.g., link, will not appear
                # Adding metadata, i.e., link, for each web content
                documents[idx].metadata['source'] = search_results[idx]['url']

            # Print a message while searching and creating a summary
            print("Searching online and creating a summary for you...")

            # Showing relevant websites
            st.write("Reading contents...")
            time.sleep(5)
            
            # default mode of building the index
            response_synthesizer = get_response_synthesizer(
                response_mode="simple_summarize", use_async=True
            )

            # Showing summarizing step
            st.write("Summarizing...")
            
            # Using summary index (much longer compared to VectorStoreIndex)
            doc_summary_index = DocumentSummaryIndex.from_documents(
                documents,
                response_synthesizer=response_synthesizer,
                show_progress=False,
            )

            # Initialize the variable to store concatenated text and metadata
            doc_summaries_and_metadata = []
            all_doc_summaries_and_metadata = ""
            
            # Loop through each element in the list
            for x in range(len(documents)):
                # Concatenate the text and metadata for the current element
                doc_summary = doc_summary_index.get_document_summary(str(x))
                doc_metadata = doc_summary_index.ref_doc_info[str(x)].metadata['source']

                current_doc_summary = f"{doc_summary}. \nSource:{doc_metadata}. \n"

                # Append the current_doc_summary to the overall variable
                doc_summaries_and_metadata.append(current_doc_summary)

                # Joining the elements with a line break
                all_doc_summaries_and_metadata = "\n".join(doc_summaries_and_metadata)

            
            # Update complete status
            status.update(label="Completed", expanded=False)

        return all_doc_summaries_and_metadata, doc_summary_index

    @staticmethod
    def new_line_after_web_link(text):
        """
        Adds a new line after each paragraph and website link.

        Args:
            text (str): The input text.

        Returns:
            str: The modified text with a new line after each paragraph and website link.
        """
        link_pattern = r"https?://[^\s]+"
        matches = re.finditer(link_pattern, text)

        for match in matches:
            website_link = match.group(0)

            if website_link:
                # Add a new line after each website link
                text = text.replace(website_link, f"{website_link}\n\n", 1)

        return text
