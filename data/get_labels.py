import os
import pandas as pd

# Path to your audio files directory
audio_dir = "all_denoised_audio_files"

# Initialize lists to store file paths and labels
file_paths = []
labels = []

# Walk through the directory and process each .wav file
for root, dirs, files in os.walk(audio_dir):
    for file in files:
        if file.endswith('.wav'):
            # Extract the label from the filename
            if '_hc_' in file:
                label = 0  # 0 for control group
            elif '_pd_' in file:
                label = 1  # 1 for patients
            else:
                continue  # skip files that don't match either pattern
            
            full_path = os.path.join(root, file)
            file_paths.append(full_path)
            labels.append(label)

# Create a DataFrame
data = {'file_path': file_paths, 'label': labels}
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("label_data.csv", sep=';', index=False)