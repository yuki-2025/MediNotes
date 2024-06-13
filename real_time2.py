import pyaudio
import wave
import torch
import json
import os
import soundfile as sf
import numpy as np
from transformers import pipeline
from resemblyzer import VoiceEncoder, preprocess_wav, sampling_rate
from pyannote.audio import Pipeline

# Set environment variable to enable CPU fallback for MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Use CPU explicitly for Whisper model
device = torch.device("cpu")

# Initialize Whisper model
whisper_model = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)

# Initialize Voice Encoder for speaker embeddings
encoder = VoiceEncoder()

# PyAudio settings
CHUNK = 1024 * 4  # Increase chunk size for better context
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = sampling_rate  # Ensure the sample rate matches what Resemblyzer expects

# Counter for JSON file names
file_counter = 1

# Buffer to accumulate audio data
audio_buffer = []
results = []

# To keep track of the last processed speaker and text to avoid repetition
last_speaker = None
last_text = ""
speaker_mapping = {"SPEAKER_00": "Doctor", "SPEAKER_01": "Patient"}

# Initialize pyannote.audio pipeline for speaker diarization
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

def transcribe_audio(audio_file):
    # Load audio
    speech, rate = sf.read(audio_file)
    input_data = {"array": np.array(speech), "sampling_rate": rate}
    
    # Perform inference with Whisper
    transcription = whisper_model(input_data)["text"]
    return transcription

def diarize_audio_segment(audio_file):
    # Perform diarization using pyannote.audio
    diarization = pipeline({"uri": "file", "audio": audio_file})
    
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append((turn.start, turn.end, speaker_mapping[speaker]))
    
    print(f"Diarization segments: {segments}")
    return segments

def save_to_json(results):
    global file_counter
    json_filename = f"rt{file_counter}.json"
    with open(json_filename, 'w') as json_file:
        json.dump(results, json_file, indent=4)
    file_counter += 1

def process_audio_in_parallel(audio_segments):
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(diarize_audio_segment, audio_segments))
    return results

def process_audio(audio_data):
    global last_speaker, last_text

    # Convert audio_data from bytes to numpy array
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

    # Save the complete audio data to a temporary file
    temp_audio_path = "temp.wav"
    with wave.open(temp_audio_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(audio_data)

    # Perform diarization on the entire audio file
    diarization_segments = diarize_audio_segment(temp_audio_path)

    all_diarization_segments = []
    for start, end, speaker in diarization_segments:
        segment_file = f"segment_{int(start * RATE)}_{int(end * RATE)}.wav"
        segment_wav = audio_array[int(start * RATE):int(end * RATE)]
        
        # Ensure the segment_wav has the correct dimensions
        if segment_wav.ndim == 1:
            segment_wav = segment_wav.reshape(-1, 1)
        
        # Write segment to file
        sf.write(segment_file, segment_wav, RATE)

        # Perform speech-to-text on each segment
        try:
            text = transcribe_audio(segment_file)
            print(f"Transcribed Text: {text}")
        except Exception as e:
            print(f"Error in transcription: {e}")
            text = ""

        if speaker != last_speaker:
            result = {
                "label": speaker,
                "words": text  # Assuming text covers the entire segment
            }
            results.append(result)
            last_speaker = speaker
            last_text = text
        all_diarization_segments.append((speaker, text))
        print("\nSpeaker Diarization:")
        for result in results:
            print(result)

    # Ensure no duplicate text segments with different labels
    unique_results = []
    seen_text = set()
    for label, words in all_diarization_segments:
        if words not in seen_text:
            seen_text.add(words)
            unique_results.append({
                "label": label,
                "words": words
            })

    results[:] = unique_results

    # Clean up temporary segment files
    for start, end, speaker in diarization_segments:
        segment_file = f"segment_{int(start * RATE)}_{int(end * RATE)}.wav"
        if os.path.exists(segment_file):
            os.remove(segment_file)

    # Remove the temporary file
    if os.path.exists("temp.wav"):
        os.remove("temp.wav")
        print("Temporary file temp.wav has been removed.")

def callback(in_data, frame_count, time_info, status):
    global audio_buffer
    audio_buffer.append(in_data)

    return (in_data, pyaudio.paContinue)

# Initialize PyAudio
p = pyaudio.PyAudio()

def main():
    global results, audio_buffer

    # Open stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    stream_callback=callback)

    input("Press Enter to start recording...")
    print("Listening...")
    stream.start_stream()

    # Keep the stream active
    try:
        while stream.is_active():
            pass
    except KeyboardInterrupt:
        print("Stopping stream...")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save and transcribe the accumulated audio data
    if audio_buffer:
        process_audio(b''.join(audio_buffer))

    # Save results to JSON
    if results:
        save_to_json(results)

    # Remove temporary segment files and temp.wav
    if os.path.exists("temp.wav"):
        os.remove("temp.wav")
        print("Temporary file temp.wav has been removed.")

if __name__ == "__main__":
    main()
