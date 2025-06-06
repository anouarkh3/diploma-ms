import math
import random
import numpy as np
from PIL import Image
import cv2
import torch
import random
import librosa
import numpy as np
import os
import librosa
import numpy as np
import pandas as pd
import glob
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Source: https://www.kaggle.com/davids1992/specaugment-quick-implementation
def spec_augment(spec: np.ndarray,
                 num_mask=2,
                 freq_masking=0.15,
                 time_masking=0.20,
                 value=0):
    spec = spec.copy()
    num_mask = random.randint(1, num_mask)
    for i in range(num_mask):
        all_freqs_num, all_frames_num  = spec.shape
        freq_percentage = random.uniform(0.0, freq_masking)

        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
        f0 = int(f0)
        spec[f0:f0 + num_freqs_to_mask, :] = value

        time_percentage = random.uniform(0.0, time_masking)

        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        spec[:, t0:t0 + num_frames_to_mask] = value
    return spec

#Source: https://github.com/lRomul/argus-freesound/blob/master/src/transforms.py
class SpecAugment:
    def __init__(self,
                 num_mask=2,
                 freq_masking=0.15,
                 time_masking=0.20):
        self.num_mask = num_mask
        self.freq_masking = freq_masking
        self.time_masking = time_masking

    def __call__(self, image):
        return spec_augment(image,
                            self.num_mask,
                            self.freq_masking,
                            self.time_masking,
                            image.min())

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, trg=None):
        if trg is None:
            for t in self.transforms:
                image = t(image)
            return image
        else:
            for t in self.transforms:
                image, trg = t(image, trg)
            return image, trg
        
class UseWithProb:
    def __init__(self, transform, prob=.5):
        self.transform = transform
        self.prob = prob

    def __call__(self, image, trg=None):
        if trg is None:
            if random.random() < self.prob:
                image = self.transform(image)
            return image
        else:
            if random.random() < self.prob:
                image, trg = self.transform(image, trg)
            return image, trg
        
class OneOf:
    def __init__(self, transforms, p=None):
        self.transforms = transforms
        self.p = p

    def __call__(self, image, trg=None):
        transform = np.random.choice(self.transforms, p=self.p)
        if trg is None:
            image = transform(image)
            return image
        else:
            image, trg = transform(image, trg)
            return image, trg
        
class ImageToTensor:
    def __call__(self, image):
        delta = librosa.feature.delta(image)
        accelerate = librosa.feature.delta(image, order=2)
        image = np.stack([image, delta, accelerate], axis=0)
        image = image.astype(np.float32) / 100
        image = torch.from_numpy(image)
        return image
    
class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, signal):
        start = random.randint(0, signal.shape[1] - self.size)
        return signal[:, start: start + self.size]

class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, signal):
        if signal.shape[1] > self.size:
            start = (signal.shape[1] - self.size) // 2
            return signal[:, start: start + self.size]
        else:
            return signal
        
class PadToSize:
    def __init__(self, size, mode='constant'):
        assert mode in ['constant', 'wrap']
        self.size = size
        self.mode = mode

    def __call__(self, signal):
        if signal.shape[1] < self.size:
            padding = self.size - signal.shape[1]
            offset = padding // 2
            pad_width = ((0, 0), (offset, padding - offset))
            if self.mode == 'constant':
                signal = np.pad(signal, pad_width,
                                'constant', constant_values=signal.min())
            else:
                signal = np.pad(signal, pad_width, 'wrap')
        return signal

def get_transforms(train, size,
                   wrap_pad_prob=0.5,
                   resize_prob=0.33,
                   spec_num_mask=2,
                   spec_freq_masking=0.15,
                   spec_time_masking=0.20,
                   spec_prob=0.5):
    if train:
        transforms = Compose([
            OneOf([
                PadToSize(size, mode='wrap'),
                PadToSize(size, mode='constant'),
            ], p=[wrap_pad_prob, 1 - wrap_pad_prob]),
            PadToSize(size),
            RandomCrop(size),
            UseWithProb(SpecAugment(num_mask=spec_num_mask,
                                    freq_masking=spec_freq_masking,
                                    time_masking=spec_time_masking), spec_prob),
            ImageToTensor()
        ])
    else:
        transforms = Compose([
            PadToSize(size),
            CenterCrop(size),
            ImageToTensor()
        ])
    return transforms

def read_as_melspectrogram(file_path):
    y, sr = librosa.load(file_path, sr=48000)
    yt, idx = librosa.effects.trim(y) 

    spectrogram = librosa.feature.melspectrogram(y=yt,
                                                 sr=48000,
                                                 n_mels=150,
                                                 hop_length=345 * 5,                                            
                                                 n_fft=315 * 20,
                                                 fmin=20,
                                                 fmax=48000//4)
                                                
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram

def predict_audio(file_path, model, device='cuda'):
    """
    Predict class for a single audio file
    
    Args:
        file_path (str): path to audio file
        model: trained model
        device: computation device ('cuda' or 'cpu')
    
    Returns:
        tuple: (probability of class 1, predicted class)
    """
    # Convert audio to mel-spectrogram
    spectrogram = read_as_melspectrogram(file_path)
    
    # Apply transforms (as for validation)
    transforms = get_transforms(train=False, size=256)
    image = transforms(spectrogram)
    
    # Add batch dimension and move to device
    image = image.unsqueeze(0).to(device)
    
    # Prediction
    with torch.no_grad():
        output = model(image)
        prob = torch.softmax(output, dim=1)[0, 1].item()  # probability of class 1
        pred_class = torch.argmax(output).item()  # predicted class
    
    return prob, pred_class
    