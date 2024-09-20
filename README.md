# Semantic Movie Recommender

This project is a **movie recommendation system** built with **FastAPI** and **Sentence Transformers**. It suggests movies based on plot similarity using pre-trained sentence embeddings.

## Overview

This project leverages **FastAPI** for building a REST API, and utilizes **Sentence Transformers** to compute embeddings for movie plots. The model employed is `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face, which provides efficient and accurate sentence embeddings. The underlying dataset used for this project is the [Wikipedia Movies dataset](https://www.kaggle.com/datasets/exactful/wikipedia-movies), containing approximately 18,000 movie plots. I used this dataset along with the pre-trained model to compute embeddings, which are stored in a pickle file located at `app/data/movie_plot_embedding_dataset.pkl`.

## Features

- FastAPI-based REST API
- Movie plot recommendations via semantic similarity
- Sentence Transformer (`all-MiniLM-L6-v2`) for embedding generation
- Scalable containerization with Docker

## Dataset

The model uses a pre-embedded movie dataset (`movie_plot_embedding_dataset.pkl`), containing movie plots and embeddings.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/omar-arif/semantic-movie-recommender.git
   cd semantic-movie-recommender
   ```
   
2. Install dependencies:
```bash
Copier le code
pip install -r requirements.txt
```

(Optional) To use Docker:
```bash
Copier le code
docker build -t semantic-movie-recommender .
docker run -p 8080:8080 semantic-movie-recommender
````

## API Endpoints
Health Check
- GET /<br />
    Returns {\"status\": \"OK\"} to verify the API is running.

Info
- GET /info <br />
    Returns API information, such as app name and description.

Movie Recommendations
- POST /recommend/<br />
    Receives a plot description and returns recommended movies based on semantic similarity.

    **Parameters**:
    - prompt: A string describing the movie plot (10-3000 characters).
    - top_k: (Optional) Number of recommendations to return (max 10).
    
    **Example Request**:<br />
        {
          \"prompt\": \"A thrilling futuristic adventure in space involving a young astronaut.\",
          \"top_k\": 5
        }

# Usage
Run the FastAPI server locally:
```bash
uvicorn app.app:app --reload --port 8080
```

# Docker
The app can be built and run using Docker:
```bash
docker build -t semantic-movie-recommender .
docker run -p 8080:8080 semantic-movie-recommender
```

# Deployment
You can access the API deployed at: (Update this with the actual deployment URL)

## License
This project is licensed under the Apache 2.0 License. See the [LICENSE](https://github.com/omar-arif/semantic-movie-recommender/blob/main/LICENSE) file for more details.

