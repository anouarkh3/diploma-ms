import argparse
import torch
from pathlib import Path
import os

from model import Model  
from data.audio_preprecessing import process_single_file
from pipeline_utils import predict_audio 

# --- FFMPEG CONFIG ---
ffmpeg_dir = Path('data/ffmpeg-7.0.2-amd64-static')
os.environ["PATH"] += os.pathsep + str(ffmpeg_dir)

from pydub import AudioSegment
AudioSegment.ffmpeg = str(ffmpeg_dir / "ffmpeg") 
AudioSegment.ffprobe = str(ffmpeg_dir / "ffprobe")

def main():
    parser = argparse.ArgumentParser(description='Audio classification pipeline with denoising and silence removal')
    parser.add_argument('--audio_path', type=str, help='Path to the raw input audio file')
    parser.add_argument('--output_dir', type=str, default='processed_audio', help='Directory to save processed audio')
    parser.add_argument('--model_path', type=str, default='model.pth', help='Path to trained model weights')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], 
                        help='Device to use for computation')

    args = parser.parse_args()

    # Prepare device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    # Load model
    model = Model().to(device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except FileNotFoundError:
        print(f"Model not found at {args.model_path}")
        return
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        return
    model.eval()

    # Preprocess audio
    processed_path = process_single_file(args.audio_path, args.output_dir)
    if not processed_path:
        print("Failed to preprocess audio.")
        return

    # Run prediction
    try:
        probability, predicted_class = predict_audio(processed_path, model, device)
        print(f"Probability of class 1 (Parkinson's Disease): {probability:.4f}")
        print(f"Predicted class: {predicted_class}")

        if predicted_class == 1:
            print("The audio is classified as Parkinson's Disease.")
        else:
            print("The audio is classified as Healthy.")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")

if __name__ == '__main__':
    main()
    