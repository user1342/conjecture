#Youtube video example of using conjecture

from youtube_transcript_api import YouTubeTranscriptApi 
from conjecture.Judge import Judge

YOUTUBE_VIDEO_ID = "wHSjrRX_eY4"
data = YouTubeTranscriptApi.get_transcript(YOUTUBE_VIDEO_ID)

judge = Judge("unsloth/mistral-7b-instruct-v0.3-bnb-4bit",data.split("."))

judge.assess()