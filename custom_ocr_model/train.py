import torch
import torch.optim as optim
from torchvision import transforms
from custom_ocr_model.modules import Model, AttnLabelConverter, IAMWordsDataset,  Averager
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# Define character set and mappings
#characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
characters = '0123456789abcdefghijklmnopqrstuvwxyz'
char_to_idx = {char: idx + 1 for idx, char in enumerate(characters)}
char_to_idx["<blank>"] = 0  # Add blank token for CTC
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

# Model Initialization
Transformation = 'None' #'TPS'
FeatureExtraction = 'ResNet'
SequenceModeling = 'BiLSTM'
Prediction = 'Attn'
num_fiducial = 20
imgH = 32
imgW = 100
input_channel = 1
output_channel = 512
hidden_size = 256
batch_max_length = 25
new_num_class = 38
batch_size = 8

model = Model(Transformation, FeatureExtraction, SequenceModeling, Prediction,
                 num_fiducial, imgH, imgW, input_channel, output_channel, hidden_size, new_num_class, batch_max_length)
model = torch.nn.DataParallel(model).to(device)
# model_path = '/content/drive/MyDrive/ArabicHandwritingRecog/best_crnn_model_2.pth'
# model.load_state_dict(torch.load(model_path, map_location=device))
# Now re-enable gradients
for param in model.parameters():
    param.requires_grad = True
    # Loss and Optimizer
#criterion = nn.CTCLoss(blank=0)
criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_avg = Averager()

# Dataset Transforms
transform = transforms.Compose([
    transforms.Resize((32, 64)),  # Resize the images to 32x64
    transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip the image horizontally
    transforms.RandomRotation(degrees=15),  # Randomly rotate the image within ±15 degrees
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),  # Apply random translation
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the image
])

# Paths to datasets
train_data = 'dataset/train'
valid_data = 'dataset/val'

train_root_dir = train_data + "/words"
train_words_txt_path = train_data + "/words.txt"

valid_root_dir = valid_data + "/words"
valid_words_txt_path = valid_data + "/words.txt"

# Create datasets and data loaders
train_dataset = IAMWordsDataset(train_root_dir, train_words_txt_path, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

valid_dataset = IAMWordsDataset(valid_root_dir, valid_words_txt_path, transform=transform)
valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

converter = AttnLabelConverter(characters)

loss_avg = Averager()


# Evaluation Function
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    ev_total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            text, length = converter.encode(labels, batch_max_length=25)
            text = text.to(device)
            length = length.to(device)

            _preds = model(images, text[:, :-1])  # align with Attention.forward
            target = text[:, 1:]  # without [GO] Symbol
            loss = criterion(_preds.view(-1, _preds.shape[-1]), target.contiguous().view(-1))

            # Compute loss
            input_lengths = torch.IntTensor([_preds.size(0)] * _preds.size(1)).to(device)
            # loss = criterion(outputs, text, input_lengths, length)
            running_loss += loss.item()

            # Decode predictions
            _, preds = _preds.max(2)
            preds = preds.transpose(1, 0)  # .contiguous().view(-1)
            decoded_preds = converter.decode(preds, input_lengths)

            list_words = []
            for pred_str, label in zip(decoded_preds, list(labels)):
                # cleaned_str = pred_str.replace("", "") #[s]
                pred_EOS = pred_str.find('[s]')
                cleaned_str = pred_str[:pred_EOS]
                list_words.append(cleaned_str)

            print(list_words)
            print(list(labels))

            ev_total_correct += sum([pred == label for pred, label in zip(list_words, list(labels))])
            total_samples += len(labels)

    if total_samples == 0:
        avg_loss = 0.0
        accuracy = 0.0
    else:
        avg_loss = running_loss / len(dataloader)
        accuracy = ev_total_correct / total_samples
        print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    return avg_loss, accuracy


def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, device):
    best_accuracy = 0.0  # Initialize the best accuracy
    tr_total_correct = 0.0
    total_samples = 0
    for epoch in range(num_epochs):
        model.train()  # Ensure the model is in training mode
        running_loss = 0.0

        for idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            text, length = converter.encode(labels, batch_max_length=25)
            text = text.to(device)

            _preds = model(images, text[:, :-1])  # align with Attention.forward
            target = text[:, 1:]  # without [GO] Symbol

            optimizer.zero_grad()  # Clear gradients from the previous step
            loss = criterion(_preds.view(-1, _preds.shape[-1]), target.contiguous().view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # Gradient clipping
            optimizer.step()

            input_lengths = torch.IntTensor([_preds.size(0)] * _preds.size(1)).to(device)

            # Decode predictions
            _, preds = _preds.max(2)
            preds = preds.transpose(1, 0)  # .contiguous().view(-1)
            decoded_preds = converter.decode(preds, input_lengths)

            list_words = []
            for pred_str, label in zip(decoded_preds, list(labels)):
                # cleaned_str = pred_str.replace("", "") #[s]
                pred_EOS = pred_str.find('[s]')
                cleaned_str = pred_str[:pred_EOS]
                list_words.append(cleaned_str)

            print(list_words)
            print(list(labels))
            tr_total_correct += sum(
                [cleaned_pred == list(labels) for cleaned_pred, label in zip(list_words, list(labels))])
            total_samples += len(labels)
            train_accuracy = tr_total_correct / total_samples

            running_loss += loss.item()

            if idx % 10 == 0:
                print(
                    f"Train : Epoch [{epoch + 1}/{num_epochs}], Batch [{idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {train_accuracy}")

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Training Loss: {epoch_loss:.4f}")

        # Validate the model at the end of the epoch
        model.eval()  # Set model to evaluation mode
        # val_loss, val_accuracy = evaluate_model(model, valid_loader, criterion, device)
        model.train()  # Set model back to training mode

        # Save the model if validation accuracy improves
        if train_accuracy >= best_accuracy:
            best_accuracy = train_accuracy
            torch.save(model.state_dict(), "/content/drive/MyDrive/ArabicHandwritingRecog/best_crnn_model_3.pth")
            print(f"Model saved with accuracy: {best_accuracy:.4f}")

    print("Training complete!")


# Train the model
num_epochs = 10
train_model(model, train_dataloader, valid_dataloader, criterion, optimizer, num_epochs, device)
