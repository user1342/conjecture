from typing import Tuple
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn.functional as F
from conjecture.support.ExtractorBaseClass import ExtractorBaseClass


class MembershipInferenceAttacks(ExtractorBaseClass):

    def _get_token_probabilities(self, input_text: str) -> Tuple[str, float]:
        """
        Gets the token probabilities for a given input sentence.

        Args:
            input_text (str): The input sentence to analyze.

        Returns:
            Tuple[str, float]: The most likely token and its probability.
        """
        inputs = self._tokenizer(input_text, return_tensors="pt")

        # Get logits from the model
        with torch.no_grad():
            outputs = self._model(**inputs)

        logits = outputs.logits

        # Softmax to convert logits into probabilities (for the last token in the sequence)
        last_token_logits = logits[0, -1, :]
        probabilities = F.softmax(last_token_logits, dim=-1)

        # Get the top 1 most likely token and its probability
        top_prob, top_index = torch.topk(probabilities, 1)
        top_token = self._tokenizer.convert_ids_to_tokens(top_index.tolist())

        return top_token[0], top_prob.item()

    def membership_inference_attack(
        self, original_text: str, variation_text: str, threshold: float = 0.1
    ) -> float:
        """
        Performs a membership inference attack by comparing the model's confidence between
        the original sentence and a slight variation. If the model's confidence on the
        original sentence is significantly higher, it suggests that the original might
        have been in the training set.

        Args:
            original_text (str): The original sentence to check.
            variation_text (str): The slight variation of the original sentence.
            threshold (float, optional): The threshold for determining if the original text
                                         is likely part of the training set. Defaults to 0.1.

        Returns:
            float: The probability difference between the original and variation text.
        """

        # Get probabilities for the original text
        original_token, original_prob = self._get_token_probabilities(original_text)

        # Get probabilities for the variation text
        variation_token, variation_prob = self._get_token_probabilities(variation_text)

        # Check the difference in probabilities
        prob_difference = original_prob - variation_prob

        # If the original sentence has much higher confidence than the variation
        if prob_difference > threshold:
            print(
                "Membership Inference Result: The original text is likely part of the training set."
            )
        else:
            print(
                "Membership Inference Result: The original text is unlikely to be part of the training set."
            )

        return prob_difference
