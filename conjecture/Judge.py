import warnings
from typing import Dict, List, Tuple

from conjecture.support.easy_llm import EasyLLM
from .extractors.AttentionPatternAnalysis import AttentionPatternAnalysis
from .extractors.MembershipInferenceAttacks import MembershipInferenceAttacks
from .extractors.QueryBasedDataExtraction import QueryBasedDataExtraction
from rich.console import Console

# Suppress warnings
warnings.filterwarnings("ignore")


class Judge:
    def __init__(self, model_name: str, list_of_dataset_strings: List[str]) -> None:
        """
        Initializes the Judge with a model and dataset strings.

        Args:
            model_name (str): The name of the model to evaluate.
            list_of_dataset_strings (List[str]): List of dataset strings for evaluation.
        """
        self._list_of_dataset_strings = list_of_dataset_strings
        self.llm = EasyLLM(model_name=model_name)
        self.llm.set_to_eval()

        self._console = Console()

    def _generate_reference_strings(
        self, topic: str, num_permutations: int
    ) -> List[str]:
        """
        Generates reference strings related to the given topic.

        Args:
            topic (str): The topic to generate reference strings for.
            num_permutations (int): The number of reference strings to generate.

        Returns:
            List[str]: A list of reference strings.
        """
        prompt = (
            f"You are an expert on the topics presented in this sentence '{topic}'. "
            f"Return {num_permutations} sentences that talk around the topics. "
            "Only return these sentences."
        )
        response = self.llm.ask_question(question=prompt, reset_dialogue=True)
        return [s.strip() for s in response.split(".") if s.strip()]

    def _generate_variation_strings(
        self, topic: str, num_permutations: int
    ) -> List[str]:
        """
        Generates variation strings based on the given topic.

        Args:
            topic (str): The topic to generate variation strings for.
            num_permutations (int): The number of variation strings to generate.

        Returns:
            List[str]: A list of variation strings.
        """
        prompt = (
            f"You are an expert on the topics presented in this sentence '{topic}'. "
            f"Return {num_permutations} sentences that are slight variations on the provided sentence. "
            "Only return these sentences."
        )
        response = self.llm.ask_question(question=prompt, reset_dialogue=True)
        return [s.strip() for s in response.split(".") if s.strip()]

    def assess_data_presence(
        apa_avg: float,
        mia_avg: float,
        qbde_avg: float,
        apa_threshold: float = 0.5,
        mia_threshold: float = 0.5,
        qbde_threshold: float = 50.0,
    ) -> str:
        """
        Assess whether the data is likely present in the dataset based on average scores
        from the three extractors and their respective thresholds.

        Args:
            apa_avg (float): Average score from the Attention Pattern Analysis.
            mia_avg (float): Average score from the Membership Inference Attacks.
            qbde_avg (float): Average score from the Query-Based Data Extraction.
            apa_threshold (float, optional): Threshold for APA score. Defaults to 0.5.
            mia_threshold (float, optional): Threshold for MIA score. Defaults to 0.5.
            qbde_threshold (float, optional): Threshold for QBDE score (percentage). Defaults to 50.0.

        Returns:
            str: Assessment result indicating whether the data is likely present in the dataset.
        """

        # Check if all scores exceed their respective thresholds
        apa_present = apa_avg > apa_threshold
        mia_present = mia_avg > mia_threshold
        qbde_present = qbde_avg > qbde_threshold

        # Determine the final assessment
        if apa_present and mia_present and qbde_present:
            return True
        else:
            return False

    def _generate_probing_strings(self, topic: str, num_permutations: int) -> List[str]:
        """
        Generates probing questions related to the given topic.

        Args:
            topic (str): The topic to generate probing questions for.
            num_permutations (int): The number of probing questions to generate.

        Returns:
            List[str]: A list of probing questions.
        """
        prompt = (
            f"You are an expert on the topics presented in this sentence '{topic}'. "
            f"Return {num_permutations} sentences that ask questions about the topics or that would get someone "
            "to answer with the sentence provided. Only return these sentences."
        )
        response = self.llm.ask_question(question=prompt, reset_dialogue=True)
        return [s.strip() for s in response.split(".") if s.strip()]

    def assess(
        self, num_permutations: int = 5
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        Assesses the model on the dataset strings using different extraction methods.

        Args:
            num_permutations (int, optional): The number of permutations for generating strings. Defaults to 5.

        Returns:
            Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
                - A dictionary with Attention Pattern Analysis scores.
                - A dictionary with Membership Inference Attacks scores.
                - A dictionary with Query Based Data Extraction scores.
        """
        model = self.llm.model
        tokenizer = self.llm.tokenizer

        apa = AttentionPatternAnalysis(model, tokenizer)
        mia = MembershipInferenceAttacks(model, tokenizer)
        qbde = QueryBasedDataExtraction(model, tokenizer)

        final_apa_scores = {}
        final_mia_scores = {}
        final_qbde_scores = {}

        with self._console.status("[bold green]Processing Entries...") as status:
            self._console.clear()
            for entry in self._list_of_dataset_strings:
                # Generate Reference Strings for Attention Pattern Analysis
                self._console.print("Generating reference strings.")
                reference_strings = self._generate_reference_strings(
                    entry, num_permutations
                )
                self._console.clear()
                self._console.print(
                    f"{len(reference_strings)} reference strings generated!"
                )

                # Generate Variation Strings for Membership Inference Attacks
                self._console.print("Generating variation strings.")
                variation_strings = self._generate_variation_strings(
                    entry, num_permutations
                )
                self._console.clear()
                self._console.print(
                    f"{len(variation_strings)} variation strings generated!"
                )

                # Generate Probing Strings for Query Based Data Extraction
                self._console.print("Generating probing strings.")
                probing_strings = self._generate_probing_strings(
                    entry, num_permutations
                )
                self._console.clear()
                self._console.print(
                    f"{len(probing_strings)} probing strings generated!"
                )

                # Attention Pattern Analysis
                apa_scores = [
                    apa.compute_likelihood(entry, ref_str)
                    for ref_str in reference_strings
                ]
                apa_score = sum(apa_scores) / len(apa_scores) if apa_scores else 0
                final_apa_scores[entry] = apa_score

                # Membership Inference Attacks
                mia_scores = [
                    mia.membership_inference_attack(entry, var_str)
                    for var_str in variation_strings
                ]
                mia_score = sum(mia_scores) / len(mia_scores) if mia_scores else 0
                final_mia_scores[entry] = mia_score

                # Query Based Data Extraction
                qbde_score = qbde.estimate_likelihood(entry, probing_strings)
                final_qbde_scores[entry] = qbde_score

                self._console.clear()
                self._console.print(
                    f"{entry}: \n MIA Score: {mia_score} \n APA Score: {apa_score} \n QDBE Score: {qbde_score}"
                )

        return final_apa_scores, final_mia_scores, final_qbde_scores
