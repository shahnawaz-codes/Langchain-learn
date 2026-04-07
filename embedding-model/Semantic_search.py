from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
load_dotenv()

# create model
embeddings = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

# documents where we perform semantic search 
documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'tell me about bumrah'

# create vector[]
query_embeddings = embeddings.embed_query(query) # returns single vectors
doc_embeddings = embeddings.embed_documents(documents)# returns list of vectors

# it taks both parameter as 2d list 
result = cosine_similarity([query_embeddings],doc_embeddings)[0]

'''
here we first sorted the List of [index ,vector] in accending order based on vector number and extract the largest one means last one element 
'''
index,vector = sorted(list(enumerate(result)),key=lambda x:x[1])[-1]


# we got related data to the query
print(documents[index])