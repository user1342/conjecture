import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Any
from conjecture.support.ExtractorBaseClass import ExtractorBaseClass

class AttentionPatternAnalysis(ExtractorBaseClass):
    def _get_attention_weights(self, model, tokenizer, text: str) -> List[torch.Tensor]:
        """
        Computes attention weights for a given text using the model.
        
        Args:
            model : The model to use for computing attention weights.
            tokenizer: The tokenizer to preprocess the text.
            text (str): The input text for which to compute attention weights.
        
        Returns:
            List[torch.Tensor]: A list of attention weights for each layer.
        """
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        attentions = outputs.attentions  # List of attention weights for each layer
        return attentions

    def _analyze_attention_patterns(self, attentions: List[torch.Tensor]) -> float:
        """
        Analyzes attention patterns to extract useful metrics.
        
        Args:
            attentions (List[torch.Tensor]): A list of attention weights for each layer.
        
        Returns:
            float: The average attention score across all layers.
        """
        layer_attentions = [attn[0].mean(dim=1).cpu().numpy() for attn in attentions]
        avg_attention = np.mean([np.mean(layer) for layer in layer_attentions])
        return avg_attention

    def _analyze_embeddings(self, model, tokenizer, text: str) -> np.ndarray:
        """
        Analyzes embeddings for a given text.
        
        Args:
            model: The model to use for computing embeddings.
            tokenizer: The tokenizer to preprocess the text.
            text (str): The input text for which to compute embeddings.
        
        Returns:
            np.ndarray: The embeddings of the input text.
        """
        inputs = tokenizer(text, return_tensors="pt")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_ids = inputs['input_ids'].to(device)
        embeddings = model.get_input_embeddings()(input_ids).detach().cpu().numpy()
        return embeddings

    def _compute_embedding_metrics(self, embedding: np.ndarray) -> Tuple[float, float]:
        """
        Computes metrics on embeddings to understand their distribution.
        
        Args:
            embedding (np.ndarray): The embeddings to analyze.
        
        Returns:
            Tuple[float, float]: Mean and standard deviation of the embeddings.
        """
        mean_embedding = np.mean(embedding)
        std_embedding = np.std(embedding)
        return mean_embedding, std_embedding

    def compute_likelihood(self, text: str, reference_text: str) -> float:
        """
        Computes the likelihood that the text was in the model's training data.
        
        Args:
            text (str): The input text to evaluate.
            reference_text (str): The reference text for comparison.
        
        Returns:
            float: A likelihood score indicating how similar the input text is to the reference text.
        """
        # Get attention weights for the text
        attentions = self._get_attention_weights(self._model, self._tokenizer, text)
        avg_attention = self._analyze_attention_patterns(attentions)
        
        # Get and analyze embeddings
        text_embedding = self._analyze_embeddings(self._model, self._tokenizer, text)
        ref_embedding = self._analyze_embeddings(self._model, self._tokenizer, reference_text)
        
        mean_text_embedding, std_text_embedding = self._compute_embedding_metrics(text_embedding)
        mean_ref_embedding, std_ref_embedding = self._compute_embedding_metrics(ref_embedding)
        
        # Compare metrics to infer likelihood
        attention_diff = abs(avg_attention - np.mean(ref_embedding))
        embedding_diff = abs(mean_text_embedding - mean_ref_embedding)
        
        # Combine metrics for a final score (customize the weightings as needed)
        likelihood_score = (1 / (1 + attention_diff)) * (1 / (1 + embedding_diff))
        
        return likelihood_score
