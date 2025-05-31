import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path
import numpy as np
import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio
from scipy.io import wavfile
from pydub import AudioSegment
from pydub.silence import split_on_silence
import warnings


def init_model():
    """Initialize and return the pretrained denoising model on GPU if available"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = pretrained.dns64().to(device)
    return model

# Global model initialization
model = init_model()

def load_audio(file_path):
    """Load audio file using torchaudio with fallback to pydub if needed"""
    try:
        # First try torchaudio
        wav, sr = torchaudio.load(file_path)
        return wav, sr
    except:
        try:
            # Fallback to pydub
            audio = AudioSegment.from_file(file_path)
            samples = np.array(audio.get_array_of_samples())
            if audio.channels > 1:
                samples = samples.reshape((-1, audio.channels))
            sr = audio.frame_rate
            wav = torch.FloatTensor(samples).t()
            return wav, sr
        except Exception as e:
            print(f"Failed to load {file_path}: {str(e)}")
            return None, None

def denoise_file(file_path, output_dir):
    """Apply denoising to a single audio file"""
    # Load audio
    wav, sr = load_audio(file_path)
    if wav is None:
        return None

    # Convert to model's expected format
    device = next(model.parameters()).device
    wav = convert_audio(wav.to(device), sr, model.sample_rate, model.chin)
    
    # Apply denoising
    with torch.no_grad():
        denoised = model(wav[None])[0]
    
    # Save processed file
    filename = Path(file_path).stem
    output_path = Path(output_dir) / f"{filename}_denoised.wav"
    wavfile.write(output_path, model.sample_rate, denoised.data.cpu().numpy().T)
    
    return str(output_path)

def remove_silence_file(file_path, output_dir):
    """Remove silent segments from an audio file"""
    try:
        # Load audio file with pydub
        sound = AudioSegment.from_file(file_path, format="wav")
        
        # Split audio on silent segments
        audio_chunks = split_on_silence(
            sound,
            min_silence_len=100,
            silence_thresh=-30,
            keep_silence=500
        )
        
        # Combine non-silent segments
        combined = AudioSegment.empty()
        for chunk in audio_chunks:
            combined += chunk
            
        # Save processed file
        filename = Path(file_path).stem
        output_path = Path(output_dir) / f"{filename}_processed.wav"
        combined.export(output_path, format="wav")
        
        return str(output_path)
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def process_single_file(file_path, output_dir):
    """Process a single audio file with denoising and silence removal"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Denoise the file
    denoised_path = denoise_file(file_path, output_dir)
    if not denoised_path:
        print(f"Failed to denoise {file_path}")
        return None
    
    # Step 2: Remove silence from denoised file
    final_path = remove_silence_file(denoised_path, output_dir)
    if not final_path:
        print(f"Failed to remove silence from {denoised_path}")
        return None
    
    # Remove intermediate denoised file
    try:
        os.remove(denoised_path)
    except Exception as e:
        print(f"Could not remove intermediate file {denoised_path}: {e}")
    
    return final_path

def process_folder(input_folder, output_folder):
    """Process all audio files in a folder"""
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all audio files in input folder
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.ogg']:
        audio_files.extend(Path(input_folder).glob(ext))
    
    if not audio_files:
        print(f"No audio files found in {input_folder}")
        return []
    
    processed_files = []
    for audio_file in audio_files:
        print(f"Processing {audio_file.name}...")
        try:
            final_path = process_single_file(str(audio_file), output_folder)
            if final_path:
                processed_files.append(final_path)
                print(f"Successfully processed {audio_file.name}")
        except Exception as e:
            print(f"Error processing {audio_file.name}: {str(e)}")
    
    return processed_files
    