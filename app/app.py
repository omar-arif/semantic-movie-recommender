from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conint, Field, field_validator
from sentence_transformers import SentenceTransformer
import pandas as pd
import app.utils as utils

# embedding model info
model_name = "sentence-transformers/all-MiniLM-L6-v2"

# load model
model = SentenceTransformer(model_name)

# load knowledge base (movie dataset with embeddings)
data = pd.read_pickle("app/data/movie_plot_embedding_dataset.pkl")

# load model
model = SentenceTransformer(model_name)

# load knowledge base (movie dataset with embeddings)
data = pd.read_pickle("app/data/movie_plot_embedding_dataset.pkl")

# create an instance of the recommender class
recommender = utils.Recommender(data, model)

# Initialize FastAPI object
app = FastAPI()

# define the message structure of the API requests and responses
class RecommendationRequest(BaseModel):

    prompt: str
    top_k: conint(le=10) = None  # Optional, maximum of 10

    @field_validator("prompt")
    def validate_prompt(cls, v):

        if not v.strip():
            raise ValueError('Prompt cannot be empty.')
        
        if len(v) < 10:
            raise ValueError('Prompt cannot be under 10 characters.')
        
        if len(v) > 3000:
            raise ValueError('Prompt cannot exceed 3000 characters.')
        
        return v

class RecommendationResponse(BaseModel):

    title: str
    plot: str
    image_url: str

@app.get("/")
async def health_check():

    return {"status": "OK"}

@app.get("/info")
async def info():

    return {
        "app_name": "Prompt Movie Recommendation API",
        "description": "This API provides movie recommendations based on plot similarity."
    }


@app.post("/recommend/", response_model=list[RecommendationResponse])
async def recommend(request: RecommendationRequest, summary="Get Movie Recommendations",
    description="""
    This endpoint receives a plot description from the user and returns a list of recommended movies
    based on the similarity of their plot embeddings. The response includes the titles, plots, and 
    images of the recommended movies.

    ### Request Parameters:
    - **prompt**: A string containing the plot description to search for. It must not be empty and 
      must not exceed 3000 characters or be under 10 characters.
    - **top_k**: An optional integer indicating the number of recommendations to return (maximum 10).
      If not provided, defaults to 5.

    ### Response:
    A list of movie recommendations, where each recommendation includes:
    - **title**: The title of the recommended movie.
    - **plot**: The plot description of the recommended movie.
    - **image**: A link to an image associated with the recommended movie.
    """,
    examples={
        "example1": {
            "summary": "Basic example",
            "value": {
                "prompt": "A thrilling futuristic adventure in space involving a young astronaut.",
                "top_k": 5
            }
        }
    }):
    
    try:
        
        # Fetch embeddings from the dataset
        embeddings = recommender.fetch_embeddings()

        # Encode the user's prompt
        query_embedding = recommender.encode_prompt(request.prompt)

        # Get top-k recommendations
        top_indices = recommender.get_topk_recom(embeddings, query_embedding, request.top_k)

        # Construct the response with title, plot, and image
        recommendations = []
        for id in top_indices:

            recommendations.append({
                "title": data.iloc[id]["title"],
                "plot": data.iloc[id]["plot"],
                "image_url": data.iloc[id]["image"]
            })

        return recommendations

    except Exception as e:
        
        raise HTTPException(status_code=500, detail=str(e))