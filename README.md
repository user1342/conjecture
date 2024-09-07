

<p align="center">
    <img width=100% src="conjecture.png">
  </a>
</p>
<p align="center"> 🤖 Evaluating the likelihood of data points in a LLM's training set 🔍 </p>
Conjecture is a tool designed to evaluate whether specific data points are likely present in a machine learning model's training dataset. It uses various methods to assess model performance and potential data leakage. At the end of the day, this is just 'conjecture'...

* **📝 Data Retrieval:** Fetch data from multiple sources, including: Wikipedia, YouTube, File, and Direct Input.
* **🔍 Model Evaluation:** Use Attention Pattern Analysis (APA), Membership Inference Attacks (MIA), and Query Based Data Extraction (QBDE) to identify the liklihood of the data existing in the models training dataset.
* **📊 Results Display:** View formatted tables of scores and average scores for each assessment category.
* **📈 Data Presence Check:** Determine if the given data is likely present in the model's training dataset based on the assessment results.

# ⚙️ Setup
## System Requirements
Conjecture is compatible with most modern systems. It requires Python 3.6 or higher.

## Dependencies
Install the required Python packages using:

```bash
pip install -r requirements.txt
```

## Installation
Clone the repository and install Conjecture:

```bash
git clone https://github.com/yourusername/conjecture.git
cd conjecture
python -m pip install .
```

# 🏃 Running
To use Conjecture, run the command with appropriate arguments:

Command-Line Arguments
* dataset_strings: List of dataset strings to evaluate.
* file: Path to a file containing newline-separated dataset strings.
* wikipedia: Wikipedia page name to fetch data from.
* youtube: YouTube video ID to fetch data from.
* model_name: Name of the model to evaluate.
* num_permutations: Number of permutations for generating strings (default: 5).

## Example Commands
Evaluate dataset strings directly:

```bash
conjecture --dataset_strings "data1" "data2" --model_name "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
```

Load dataset from a file:

```bash
conjecture --file "path/to/dataset.txt" --model_name "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
```

Fetch data from Wikipedia:

```bash
conjecture --wikipedia "Machine_learning" --model_name "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
```
Fetch transcript from YouTube:

```bash
conjecture --youtube "wHSjrRX_eY4" --model_name "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
```
## Programmatic Examples
### YouTube Video Example

Fetches the transcript of a YouTube video and assesses whether the data is present in the model's training dataset:

```python
from youtube_transcript_api import YouTubeTranscriptApi
from conjecture.Judge import Judge

YOUTUBE_VIDEO_ID = "wHSjrRX_eY4"
data = YouTubeTranscriptApi.get_transcript(YOUTUBE_VIDEO_ID)

judge = Judge("unsloth/mistral-7b-instruct-v0.3-bnb-4bit", data.split("."))
judge.assess()
```

### Wikipedia Example

Fetches introductory text from a Wikipedia page and assesses whether the data is present in the model's training dataset:

```python
import requests
from conjecture.Judge import Judge

WIKIPEDIA_PAGE_NAME = "YouTube"

response = requests.get(
    'https://en.wikipedia.org/w/api.php',
    params={
        'action': 'query',
        'format': 'json',
        'titles': WIKIPEDIA_PAGE_NAME,
        'prop': 'extracts',
        'exintro': True,
        'explaintext': True,
    }
).json()

page = next(iter(response['query']['pages'].values()))
data = page['extract']

judge = Judge("unsloth/mistral-7b-instruct-v0.3-bnb-4bit", data.split("."))
judge.assess()
```

# 🔍 Output
Conjecture will display the following:

* Tables of Scores: APA, MIA, and QBDE scores along with their averages.
* Data Presence Check: A message indicating whether the data is likely present in the model's training dataset.

# 🙏 Contributions
Conjecture is open-source and welcomes contributions. To contribute:

* Fork the repository on GitHub.
* Create a new branch for your changes.
* Implement and test your changes.
* Submit a pull request with a clear description.

# 🐛 Issues and Feature Requests
Report bugs or request features by opening an issue on GitHub. Provide detailed information to assist in addressing your concerns.

# 📜 License
GNU General Public License v3.0

