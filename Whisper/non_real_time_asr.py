import os
import json
from pydub import AudioSegment
import wave
from pyannote.audio import Pipeline
from transformers import pipeline
import torch
from huggingface_hub import login
from multiprocessing import Pool, cpu_count
import numpy as np
import signal

# Set environment variable to enable CPU fallback for MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Timeout handler
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Diarization process timed out")

# Register the timeout handler
signal.signal(signal.SIGALRM, timeout_handler)

def convert_m4a_to_wav(input_path, output_path):
    try:
        print(f"Converting {input_path} to {output_path}")
        audio = AudioSegment.from_file(input_path, format="m4a")
        audio.export(output_path, format="wav")
        print(f"Conversion complete: {output_path}")
    except Exception as e:
        print(f"Error during conversion: {e}")

def transcribe_audio_file(filepath, whisper_model):
    try:
        # Load audio data
        with wave.open(filepath, "rb") as wf:
            data = wf.readframes(wf.getnframes())
            rate = wf.getframerate()
            array = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0  # Convert to float32 and normalize
            input_data = {"array": array, "sampling_rate": rate}
            # Perform transcription
            transcription = whisper_model(input_data)["text"]
            words = [{"word": w, "start": i, "end": i + 1} for i, w in enumerate(transcription.split())]  # Mock word timings
            return transcription, words
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        return None, None

def diarize_audio_file(filepath, pipeline, device, timeout_duration=600):  # Increased timeout duration
    try:
        print(f"Diarizing file: {filepath}")
        signal.alarm(timeout_duration)  # Set the timeout
        pipeline.to(device)  # Ensure the pipeline is using the correct device
        diarization = pipeline({'audio': filepath})
        signal.alarm(0)  # Cancel the timeout
        return diarization
    except TimeoutException:
        print(f"Diarization for {filepath} timed out.")
        return None
    except Exception as e:
        print(f"Error during diarization: {e}")
        return None

def process_file(file_info):
    filename, directory, hf_token, device = file_info
    transcriptions = {}
    diarizations = {}

    if filename.endswith(".m4a"):
        filepath = os.path.join(directory, filename)
        wav_filename = filename.replace(".m4a", ".wav")
        wav_filepath = os.path.join(directory, wav_filename)

        print(f"Found m4a file: {filepath}")
        convert_m4a_to_wav(filepath, wav_filepath)

        print(f"Processing file: {wav_filepath}")

        # Initialize Whisper model
        whisper_model = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)

        transcription, word_list = transcribe_audio_file(wav_filepath, whisper_model)
        if transcription:
            transcriptions[wav_filename] = (transcription, word_list)
            print(f"Transcription for {wav_filepath}: {transcription}")

        # Initialize the Hugging Face login and diarization pipeline within each process
        login(token=hf_token)
        try:
            diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)
            diarization = diarize_audio_file(wav_filepath, diarization_pipeline, device)
            if diarization:
                diarizations[wav_filename] = diarization
                print(f"Diarization for {wav_filepath}: {diarization}")
        except Exception as e:
            print(f"Error during diarization: {e}")

    return transcriptions, diarizations

def transcribe_and_diarize_audio_files(directory, hf_token):
    transcriptions = {}
    diarizations = {}
    print(f"Directory to scan: {directory}")

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    file_info = [(filename, directory, hf_token, device) for filename in os.listdir(directory)]

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_file, file_info)

    for result in results:
        trans, diar = result
        transcriptions.update(trans)
        diarizations.update(diar)

    return transcriptions, diarizations

def label_speakers(diarization):
    speakers = {}
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speakers:
            speakers[speaker] = "Doctor" if len(speakers) == 0 else "Patient"
    return speakers

def format_output(transcription, word_list, diarization, speakers):
    diarized_transcript = []
    diarized_segments = []

    segments = [(segment, speakers[speaker]) for segment, _, speaker in diarization.itertracks(yield_label=True)]
    labeled_segments = [(segment.start, segment.end, label) for segment, label in segments]
    labeled_segments.sort()  # Ensure segments are sorted by start time

    current_speaker = None
    current_text = []

    for segment in labeled_segments:
        start, end, label = segment
        words_in_segment = [word['word'] for word in word_list if start <= word['start'] < end]
        if words_in_segment:
            if current_speaker != label:
                if current_speaker is not None:
                    diarized_transcript.append(f"[{current_speaker}] {' '.join(current_text)}")
                current_speaker = label
                current_text = words_in_segment
            else:
                current_text.extend(words_in_segment)
            diarized_segments.append({
                "start": start,
                "end": end,
                "label": label,
                "words": ' '.join(words_in_segment)
            })
    if current_text:
        diarized_transcript.append(f"[{current_speaker}] {' '.join(current_text)}")

    return '\n'.join(diarized_transcript), diarized_segments

def save_to_json(transcriptions, diarizations, output_directory):
    output = {}
    diarized_output = {}
    for filename in transcriptions:
        transcription, word_list = transcriptions[filename]
        diarization = diarizations.get(filename)
        if diarization:
            speakers = label_speakers(diarization)
            formatted_output, diarized_segments = format_output(transcription, word_list, diarization, speakers)
            output[filename] = {
                "transcription": transcription,
                "diarization": formatted_output
            }
            diarized_output[filename] = {
                "diarized_text": formatted_output,
                "segments": diarized_segments
            }
        else:
            output[filename] = {
                "transcription": transcription
            }

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    output_filepath = os.path.join(output_directory, "output.json")
    with open(output_filepath, 'w') as f:
        json.dump(output, f, indent=4)
    print(f"Results saved to {output_filepath}")

    diarized_output_filepath = os.path.join(output_directory, "diarized_output.json")
    with open(diarized_output_filepath, 'w') as f:
        json.dump(diarized_output, f, indent=4)
    print(f"Diarized segments saved to {diarized_output_filepath}")

if __name__ == "__main__":
    # Authenticate with Hugging Face
    hf_token = "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"
    login(token=hf_token)

    audio_directory = os.path.join(os.path.dirname(__file__), '..', 'MediNotes', 'Recordings')
    output_directory = os.path.join(os.path.dirname(__file__), '..', 'MediNotes', 'Output')
    print(f"Audio directory: {audio_directory}")

    transcriptions, diarizations = transcribe_and_diarize_audio_files(audio_directory, hf_token)
    
    save_to_json(transcriptions, diarizations, output_directory)
