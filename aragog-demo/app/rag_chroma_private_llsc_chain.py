# Imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.documents import Document
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, chain
from langchain_text_splitters import RecursiveCharacterTextSplitter

from typing import Any, Dict, Iterator, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk

import requests
import json
import os
import glob

# Classes
class CustomLLM(LLM):
    """
    A custom chat model that interfaces with a vLLM server running on a compute node.
    """

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input.

        Args:
            prompt: The prompt to generate from.
            stop: Not used. Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.
            run_manager: Not used. Callback manager for the run.
            **kwargs: Not used. Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            The model output as a string. Actual completions SHOULD NOT include the prompt.
        """

        # Set up header
        headers = {
                "Content-Type": "application/json",
                }

        # Setup input to model
        data = {
            "prompt": prompt,
            "max_tokens": 200,
            "temperature": 0.2,
            "n": 1,
            "best_of": 10,
            "top_p": 0.95,
            "presence_penalty": 0.2
           }

        # Send request to server
        response = requests.post('http://127.0.0.1:8080/generate', headers=headers, json=data)

        # Format response
        responseText = json.loads(response.text)["text"]
        responseText = responseText[0].split("Question: ")[1]
        responseText = responseText.split("<|eot_id|>")[0]

        return responseText

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "custom"

# Functions
def get_documents(pdfDirPath):
    """
    Get all PDF documents from path.
    """

    # Variables
    outList = []

    # Get list of PDF paths
    pathList = glob.glob(pdfDirPath+"/*.pdf")

    print("PATH: {}".format(pdfDirPath))
    print("LIST OF PAPERS: {}".format(pathList))

    # Iterate through PDFs
    for path in pathList:
        # Load PDF
        loader = PyPDFLoader(path)
        pages = loader.load_and_split()

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = text_splitter.split_documents(pages)

        # Store documents
        outList += all_splits

    return outList



# Main script #
# Load data
pdfDirPath = os.environ['PDFDIRPATH']
docList = get_documents(pdfDirPath)

# Set up embedding model
embeddingModelPath = os.environ['EMBEDMODELPATH']
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddingModel = HuggingFaceEmbeddings(
    model_name=embeddingModelPath,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=docList,
    collection_name="llsc_paper",
    embedding=embeddingModel,
)
#retriever = vectorstore.as_retriever()

@chain
def retriever(query: str) -> List[Document]:
    docs, scores = zip(*vectorstore.similarity_search_with_score(query))
    for doc, score in zip(docs, scores):
        doc.metadata["score"] = score

    return docs

# Prompt template
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Set up vLLM integration for LLM
model = CustomLLM()

# RAG chain
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)

# Add typing for input
class Question(BaseModel):
    __root__: str

chain = chain.with_types(input_type=Question)
