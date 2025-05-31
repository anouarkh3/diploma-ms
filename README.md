# Audio Classification Pipeline

This project is an audio classification pipeline designed to classify audio files for the detection of Parkinson's Disease. It includes audio preprocessing steps such as denoising and silence removal, followed by prediction using a trained machine learning model.

## Features

- **Audio Preprocessing**: Removes noise and silence from audio files
- **Model Inference**: Loads a pre-trained model to classify audio files
- **Device Compatibility**: Supports both CUDA-enabled GPUs and CPUs for computation

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Download and place the ffmpeg binaries in the `data/ffmpeg-7.0.2-amd64-static` directory.

## Usage

To run the audio classification pipeline, use the following command:

```bash
poetry run python final_pipeline.py \
  --audio_path <path_to_audio_file> \
  [--output_dir <output_directory>] \
  [--model_path <model_weights>] \
  [--device <cuda|cpu>]
```

### Parameters

| Parameter      | Description                                      | Default Value       |
|----------------|--------------------------------------------------|---------------------|
| `--audio_path` | Path to the raw input audio file (required)      | -                   |
| `--output_dir` | Directory to save processed audio               | `processed_audio`   |
| `--model_path` | Path to trained model weights                   | `model.pth`         |
| `--device`     | Device to use for computation (`cuda` or `cpu`) | `cuda`              |

## Example

To classify an audio file located at `data/all_audio_files/ID02_pd_2_0_0_1.wav`, run:

```bash
poetry run python final_pipeline.py \
  --audio_path data/all_audio_files/ID02_pd_2_0_0_1.wav
```

## Model

The model used for prediction is defined in the `model.py` file. Ensure that you have a trained model saved as `model.pth` or specify a different path using the `--model_path` argument.
```
