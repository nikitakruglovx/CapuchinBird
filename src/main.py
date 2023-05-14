import os
import librosa
import torch
import torch.nn as nn

from torchvision import models
from skimage import io
from torchvision import transforms
from matplotlib import pyplot as plt
from config import num_features

def ProcessAudio(audio_file, transform):
    y, sr = librosa.load(audio_file, mono=True, duration=5)
    plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap='inferno', sides="default", mode="default", scale="dB")
    plt.axis("off")
    plt.savefig("data.png")
    plt.clf()

    image = io.imread("data.png")
    image = transform(image)

    return image.unsqueeze(0)

def PredictionAudio(audio_file, net):
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: x[:, :, :3]),  # оставляем только 3 канала
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(244),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = ProcessAudio(audio_file, transform)
    output = net(image)

    _, pred = torch.max(output.data, 1)

    return "Capuchine Bird!" if pred.item() == 0 else "Not Capuchine Bird!"

if __name__ == "__main__":
    model = input("Select model .pt: ")

    while True:
        audio_path = input("Audio path: ")

        net = models.resnet18(pretrained=True)
        net.fc = nn.Linear(num_features, 2)
        net.load_state_dict(torch.load(model))
        net.eval()

        print("Результат предсказания:", PredictionAudio(audio_path, net))

        os.remove("data.png")

