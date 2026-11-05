from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

load_dotenv()

client = OpenAI()

# Vector Embeddings
embedding_models = OpenAIEmbeddings(model="text-embedding-3-large")

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://vector-db:6333",
    collection_name="RAG_Project",
    embedding=embedding_models,
)


# Take User Query

query = input("> ")


# Vector similarity search [query] in DB

search_results = vector_db.similarity_search(query=query)

context = "\n\n\n".join(
    [
        f"page content: {result.page_content}\n Page Number: {result.metadata['page_label']} \n File Location: {result.metadata['source']}"
        for result in search_results
    ]
)


SYSTEM_PROMPT = f"""

You are a helpful AI assistant who answers user query based on available context 
retrieved from a PDF file with page_contents along with page number.

You should only answer the user based on the following context and navigate the user to open the right page number to know more.

  Context:
   {context}



"""

chat_completion = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ],
)

print(f"🤖:  {chat_completion.choices[0].message.content}")
