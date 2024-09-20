import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

class Recommender():
    """
    A class that provides movie recommendations based on plot similarity.

    Attributes:
        df (pd.DataFrame): A DataFrame containing movie data, including titles, plots, movie image, and plot embeddings.
        model : A pre-trained model used to encode plot descriptions into embeddings.
    """

    def __init__(self, df: pd.DataFrame, model: SentenceTransformer):
        
        self.df = df
        self.model = model

    def fetch_embeddings(self) -> torch.Tensor:
        """
        Fetches plot embeddings from the DataFrame.

        Returns:
            torch.Tensor: A tensor containing the plot embeddings for all movies.
        """
        
        plot_embeddings = self.df["plot_embedding"]
        plot_embeddings = np.vstack(plot_embeddings.to_numpy()).astype(np.float32)
        return torch.from_numpy(plot_embeddings)

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """
        Encodes a user-provided plot description into an embedding.

        """

        return self.model.encode(prompt, convert_to_tensor=True)

    def get_similarity_scores(self, indexes: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """
        Calculates cosine similarity scores between the provided embeddings and a query embedding.

        Parameters:
            indexes : The embeddings of the movie plots.
            query : The embedding of the user-provided plot description.

        Returns:
             A tensor containing similarity scores for each movie plot.
        """
        
        return torch.nn.functional.cosine_similarity(indexes.unsqueeze(1), query.unsqueeze(0), dim=-1)

    def get_topk_recom(self, indexes: torch.Tensor, query: torch.Tensor, k: int = 3) -> list[int]:
        """
        Retrieves the indices of the top-k most similar movie plots to the provided query embedding.
        
        """
        
        scores = self.get_similarity_scores(indexes, query)
        top_indices = torch.topk(scores, k, dim=0).indices
        return [k.item() for k in top_indices]

