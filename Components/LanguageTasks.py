import google.generativeai as genai
from dotenv import load_dotenv
import os
import json

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API"))

if not os.getenv("GEMINI_API"):
    raise ValueError("GEMINI_API key not found. Make sure it is defined in the .env file.")

# Function to extract start and end times
def extract_times(json_string):
    try:
        data = json.loads(json_string)
        start_time = float(data[0]["start"])
        end_time = float(data[0]["end"])
        return int(start_time), int(end_time)
    except Exception as e:
        print(f"Error in extract_times: {e}")
        return 0, 0

system = """
Baised on the Transcription user provides with start and end, Highilight the main parts in less then 1 min which can be directly converted into a short. highlight it such that its intresting and also keep the time staps for the clip to start and end. only select a continues Part of the video

Follow this Format and return in valid json 
[{
start: "Start time of the clip",
content: "Highlight Text",
end: "End Time for the highlighted clip"
}]
it should be one continues clip as it will then be cut from the video and uploaded as a tiktok video. so only have one start, end and content

Dont say anything else, just return Proper Json. no explanation etc

IF YOU DONT HAVE ONE start AND end WHICH IS FOR THE LENGTH OF THE ENTIRE HIGHLIGHT, THEN 10 KITTENS WILL DIE, I WILL DO JSON['start'] AND IF IT DOESNT WORK THEN...
"""

User = """
Any Example
"""

def GetHighlight(Transcription):
    print("Getting Highlight from Transcription")
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")

        # In Gemini, there's no separate "system" role, so we prepend it
        prompt = system + "\n\n" + Transcription

        response = model.generate_content(prompt)

        json_string = response.text
        json_string = json_string.replace("json", "").replace("```", "")
        
        Start, End = extract_times(json_string)
        if Start == End:
            Ask = input("Error - Get Highlights again (y/n) -> ").lower()
            if Ask == "y":
                Start, End = GetHighlight(Transcription)
        return Start, End
    except Exception as e:
        print(f"Error in GetHighlight: {e}")
        return 0, 0

if __name__ == "__main__":
    print(GetHighlight(User))
