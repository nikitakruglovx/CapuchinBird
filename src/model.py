import os
import librosa
import torch
import torch.nn as nn
import torch.optim as optim

from skimage import io
from config import num_features, lr, momentum, epoche
from torchvision import transforms, models
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Global Param
labels = ["Parsed_Capuchinbird_Clips", "Parsed_Not_Capuchinbird_Clips"]
CAPUCHINE_FILE = "../Data/Parsed_Capuchinbird_Clips"
NOT_CAPUCHINE_FILE = "../Data/Parsed_Not_Capuchinbird_Clips"
data_list = []
answer_list = []
dictonary = {}

# Create Dataset
class CreateAudioDataset(Dataset):
    def __init__(self, datas, answer_dict, transform):
        self.datas = datas
        self.answer_dict = answer_dict
        self.transform = transform

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        image = io.imread(self.datas[idx])
        image = self.transform(image)
        label = self.answer_dict[idx]
        return image, label

# Create spectrogram for all audio files
def CreateSpectrogram():
    for i, c in enumerate(labels, 0):
        dictonary[i] = c
        dictonary[c] = i

    for folder in ["Parsed_Capuchinbird_Clips", "Parsed_Not_Capuchinbird_Clips"]:
        for file_name in os.listdir("../Data/" + folder):
            y, sr = librosa.load("../Data/" + folder + '/' + file_name, mono=True, duration=5)
            # Fs - sampling frequency
            # NFFT - size block data, using in FFT
            plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=plt.get_cmap('inferno'), sides='default',
                         mode='default', scale='dB')
            plt.axis('off')
            plt.savefig("../images/" + folder + file_name[:-3] + "png")
            plt.clf()
            data_list.append("../images/" + folder + file_name[:-3] + "png")
            answer_list.append(dictonary[folder])


def DivideDataset():
    # Divide datas on train and test
    file_train, file_test, answer_train, answer_test = train_test_split(data_list, answer_list, test_size=0.3)

    # Resize img
    transform = transforms.Compose(
        [transforms.Lambda(lambda x: x[:, :, :3]),
         transforms.ToPILImage(),
         transforms.Resize(256),
         transforms.CenterCrop(244),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    trainset = CreateAudioDataset(file_train, answer_train, transform)
    testset = CreateAudioDataset(file_test, answer_test, transform)

    # Data loader
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True)
    testloader = DataLoader(testset, batch_size=4, shuffle=True)

    return trainloader, testloader

def train():
    trainloader = DivideDataset()[0]
    testloader = DivideDataset()[1]

    net = models.resnet18(pretrained=True)
    net.fc = nn.Linear(num_features, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    losses = []
    running_corrects = 0

    net.train(True)

    # Train neuronetwork
    for epoch in range(epoche):
        running_loss = 0.0
        running_corrects = 0.0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()
            outputs = net(inputs)
            _, preds = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_corrects += int(torch.sum(preds == labels.data)) / len(labels)

            if i % 10 == 9:
                print('[%d, %5d] loss: %.3f accuracy: %.3f' % (
                epoch + 1, i + 1, running_loss / 10, running_corrects / 10))
                losses += [running_loss / 10]
                running_loss = 0.0
                running_corrects = 0.0

    net.train(False)
    print('Model train complite')
    runninig_correct = 0
    num_of_tests = 0

    # Test model
    for data in testloader:
        inputs, labels = data
        output = net(inputs)
        _, predicted = torch.max(output, 1)
        runninig_correct += int(torch.sum(predicted == labels)) / len(labels)
        num_of_tests += 1

    print('Score:', runninig_correct / num_of_tests)
    torch.save(net.state_dict(), '../model/capuchinebird.pt')

if __name__ == "__main__":
    CreateSpectrogram()
    train()



