import argparse
import requests
from typing import Dict, List, Union
from youtube_transcript_api import YouTubeTranscriptApi
from .Judge import Judge
from rich.console import Console
from rich.table import Table


def fetch_wikipedia_data(page_name: str) -> List[str]:
    """
    Fetches introductory text from a Wikipedia page.

    Args:
        page_name (str): The title of the Wikipedia page.

    Returns:
        List[str]: A list of strings, each representing a line from the introductory text.
    """
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": page_name,
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
        },
    ).json()

    page = next(iter(response["query"]["pages"].values()))
    return page["extract"].splitlines()


def fetch_youtube_data(video_id: str) -> List[str]:
    """
    Fetches transcript text from a YouTube video.

    Args:
        video_id (str): The ID of the YouTube video.

    Returns:
        List[str]: A list of strings, each representing a line from the transcript.
    """
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return [entry["text"] for entry in transcript]


def read_file_data(file_path: str) -> List[str]:
    """
    Reads lines from a file and returns them as a list of strings.

    Args:
        file_path (str): The path to the file.

    Returns:
        List[str]: A list of lines from the file.
    """
    with open(file_path, "r") as file:
        return file.read().splitlines()


def calculate_average(scores: List[float]) -> float:
    """
    Calculates the average of a list of scores.

    Args:
        scores (List[float]): A list of numeric scores.

    Returns:
        float: The average score. Returns 0 if the list is empty.
    """
    return sum(scores) / len(scores) if scores else 0


def print_scores(title: str, scores: Dict[str, float], average: float) -> None:
    """
    Prints a formatted table of scores and the average score.

    Args:
        title (str): The title of the table.
        scores (Dict[str, float]): A dictionary with entries and their corresponding scores.
        average (float): The average score to display.
    """
    console = Console()

    table = Table(title=title)
    table.add_column("Entry", style="bold")
    table.add_column("Score", style="dim")

    for entry, score in scores.items():
        table.add_row(entry, f"{score:.2f}")

    console.print(table)
    console.print(f"\nAverage Score: [bold cyan]{average:.2f}[/bold cyan]")


def main() -> None:
    """
    Main function to parse arguments, fetch data, assess model performance, and print results.
    """
    parser = argparse.ArgumentParser(
        description="Assess model performance on datasets."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dataset_strings",
        type=str,
        nargs="+",
        help="List of dataset strings to evaluate.",
    )
    group.add_argument(
        "--file",
        type=str,
        help="Path to a file containing new line separated dataset strings.",
    )
    group.add_argument(
        "--wikipedia", type=str, help="Wikipedia page name to fetch data from."
    )
    group.add_argument(
        "--youtube", type=str, help="YouTube video ID to fetch data from."
    )

    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the model to evaluate."
    )
    parser.add_argument(
        "--num_permutations",
        type=int,
        default=5,
        help="Number of permutations for generating strings.",
    )

    args = parser.parse_args()

    # Fetch data based on provided arguments
    if args.dataset_strings:
        dataset_strings = args.dataset_strings
    elif args.file:
        dataset_strings = read_file_data(args.file)
    elif args.wikipedia:
        dataset_strings = fetch_wikipedia_data(args.wikipedia)
    elif args.youtube:
        dataset_strings = fetch_youtube_data(args.youtube)
    else:
        raise ValueError("No valid input source provided.")

    if len(dataset_strings) <= 0:
        raise Exception("No valid dataset genearted or provided!")

    # Instantiate Judge
    judge = Judge(model_name=args.model_name, list_of_dataset_strings=dataset_strings)

    # Perform assessment
    apa_scores, mia_scores, qbde_scores = judge.assess(
        num_permutations=args.num_permutations
    )

    # Calculate averages
    apa_avg = calculate_average(list(apa_scores.values()))
    mia_avg = calculate_average(list(mia_scores.values()))
    qbde_avg = calculate_average(list(qbde_scores.values()))

    # Print results using rich
    print_scores("Attention Pattern Analysis Scores", apa_scores, apa_avg)
    print_scores("Membership Inference Attacks Scores", mia_scores, mia_avg)
    print_scores("Query Based Data Extraction Scores", qbde_scores, qbde_avg)

    console = Console()
    if judge.assess_data_presence(apa_avg, mia_avg, qbde_avg):
        console.print("Data is liklely present in the models training dataset")
    else:
        console.print("Data is not liklely present in the models training dataset")


if __name__ == "__main__":
    main()
