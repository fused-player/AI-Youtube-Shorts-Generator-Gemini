import subprocess
import os
import re

def transcribeAudio(
    audio_path,
    whisper_binary_path="/home/pani/Projects/AutoShorts/whisper.cpp/build/bin/whisper-cli",
    model_path="/home/pani/Projects/AutoShorts/whisper.cpp/models/ggml-base.bin"
):
    try:
        print("Transcribing audio...")
        # Check if audio file exists
        if not os.path.exists(audio_path):
            print(f"Error: Audio file {audio_path} does not exist")
            return []

        # Check if whisper.cpp binary and model exist
        if not os.path.exists(whisper_binary_path):
            print(f"Error: whisper.cpp binary {whisper_binary_path} not found")
            return []
        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} not found")
            return []

        # Construct whisper.cpp command
        base_name = os.path.splitext(audio_path)[0]  # e.g., "audio" from "audio.wav"
        output_base = os.path.join(os.path.dirname(audio_path), base_name)  # e.g., "/path/to/audio"
        command = [
            whisper_binary_path,
            "-m", model_path,
            "-f", audio_path,
            "-osrt",
            "--language", "auto",
            "--output-file", output_base  # Specify base name for SRT output
        ]

        print(f"Running whisper.cpp command: {' '.join(command)}")
        # Run whisper.cpp
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )

        # Print stderr for debugging
        if result.stderr:
            print(f"whisper.cpp stderr: {result.stderr}")

        # Check if SRT file was created
        output_srt = output_base + ".srt"  # e.g., "/path/to/audio.srt"
        if not os.path.exists(output_srt):
            print(f"Error: SRT file {output_srt} was not created")
            return []

        # Parse SRT file
        extracted_texts = []
        with open(output_srt, "r", encoding="utf-8") as f:
            srt_content = f.read()
            print(f"Parsing {output_srt} content:\n{srt_content}")  # Debug the content
            blocks = srt_content.strip().split("\n\n")

            for block in blocks:
                lines = block.strip().split("\n")
                if len(lines) >= 3:
                    # Skip empty blocks
                    if not block.strip():
                        continue
                        
                    # Extract index
                    index_match = re.match(r"(\d+)", lines[0])
                    if index_match:
                        # Extract timestamp and text
                        if len(lines) >= 2:
                            timestamp_line = lines[1]
                            text_lines = lines[2:]
                            match = re.match(r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})", timestamp_line)
                            if match:
                                start_time, end_time = match.groups()
                                text = "\n".join(text_lines).strip()  # Handle multi-line text
                                
                                def time_to_seconds(time_str):
                                    # Split on colon first, then handle milliseconds separately
                                    time_parts = time_str.split(":")
                                    hours = int(time_parts[0])
                                    minutes = int(time_parts[1])
                                    # Handle seconds and milliseconds (format: SS,mmm)
                                    seconds_part = time_parts[2]
                                    if "," in seconds_part:
                                        seconds, milliseconds = seconds_part.split(",")
                                        total_seconds = float(seconds) + float(milliseconds) / 1000.0
                                    else:
                                        total_seconds = float(seconds_part)
                                    return hours * 3600 + minutes * 60 + total_seconds

                                start = time_to_seconds(start_time)
                                end = time_to_seconds(end_time)
                                # Fix: Ensure we're appending exactly 3 values as a tuple/list
                                extracted_texts.append((text, start, end))
                            else:
                                print(f"Invalid timestamp format in block:\n{block}")
                        else:
                            print(f"Insufficient lines in block:\n{block}")
                    else:
                        print(f"No index found in block:\n{block}")

        # Clean up SRT file
        if os.path.exists(output_srt):
            os.remove(output_srt)

        if not extracted_texts:
            print("No transcriptions found in SRT output")
        return extracted_texts

    except subprocess.CalledProcessError as e:
        print(f"Transcription Error: whisper.cpp failed with exit code {e.returncode}")
        print(f"Error output: {e.stderr}")
        return []
    except Exception as e:
        print(f"Transcription Error: {str(e)}")
        import traceback
        traceback.print_exc()  # This will help debug the actual error
        return []

if __name__ == "__main__":
    audio_path = "audio.wav"  # Match your current usage
    transcriptions = transcribeAudio(audio_path)
    print("Done")
    TransText = ""

    if transcriptions:
        for text, start, end in transcriptions:
            TransText += f"{start:.2f} - {end:.2f}: {text}\n"
        print(TransText)
    else:
        print("No transcriptions to display")