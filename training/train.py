import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# --- 1. Define the Neural Network Architecture ---
class DigitClassifier(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) for digit classification.
    It consists of two convolutional layers followed by two fully connected layers.
    """
    def __init__(self):
        super(DigitClassifier, self).__init__()
        # First convolutional layer: input channels=1 (grayscale image), output channels=32, kernel size=3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Second convolutional layer: input channels=32, output channels=64, kernel size=3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Max pooling layer: reduces spatial dimensions by half
        self.pool = nn.MaxPool2d(2, 2)
        # Dropout layer: helps prevent overfitting by randomly setting a fraction of input units to 0 at each update
        self.dropout1 = nn.Dropout(0.25)

        # Fully connected layers
        # The input size to the first fully connected layer depends on the output size of the last pooling layer.
        # For a 28x28 MNIST image, after two conv+pool layers, the size becomes 7x7.
        # So, 64 (output channels from conv2) * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout2 = nn.Dropout(0.5)
        # Output layer: 128 input features, 10 output features (for 10 digits 0-9)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """
        Defines the forward pass of the network.
        """
        # Apply conv1 -> ReLU -> pool
        x = self.pool(torch.relu(self.conv1(x)))
        # Apply conv2 -> ReLU -> pool
        x = self.pool(torch.relu(self.conv2(x)))
        # Apply dropout after the second pooling layer
        x = self.dropout1(x)
        # Flatten the output for the fully connected layers
        # x.size(0) is the batch size
        x = x.view(-1, 64 * 7 * 7)
        # Apply fc1 -> ReLU -> dropout
        x = self.dropout2(torch.relu(self.fc1(x)))
        # Apply the final fully connected layer
        x = self.fc2(x)
        return x

# --- 2. Load and Preprocess the MNIST Dataset ---
def load_mnist_data(batch_size=64):
    """
    Loads the MNIST dataset, applies transformations, and creates data loaders.
    """
    # Define transformations to apply to the images
    # ToTensor converts PIL Image or numpy.ndarray to FloatTensor and scales the image pixels to [0.0, 1.0]
    # Normalize with mean and standard deviation for MNIST (calculated over the entire dataset)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST mean and std
    ])

    # Download and load the training dataset
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    # Create a DataLoader for the training set
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    # Download and load the test dataset
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    # Create a DataLoader for the test set
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader

# --- 3. Training Function ---
def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    """
    Trains the neural network model.
    """
    model.train() # Set the model to training mode
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data and target to the specified device (CPU or GPU)
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad() # Zero the gradients before each backward pass
            output = model(data) # Forward pass
            loss = criterion(output, target) # Calculate the loss
            loss.backward() # Backward pass: compute gradient of the loss with respect to model parameters
            optimizer.step() # Update model parameters

            running_loss += loss.item()
            if batch_idx % 100 == 99: # Print every 100 batches
                print(f'Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], '
                      f'Loss: {running_loss / 100:.4f}')
                running_loss = 0.0
    print('Finished Training')

# --- 4. Evaluation Function ---
def evaluate_model(model, test_loader, device):
    """
    Evaluates the trained neural network model on the test set.
    """
    model.eval() # Set the model to evaluation mode (disables dropout, batch normalization updates)
    correct = 0
    total = 0
    with torch.no_grad(): # Disable gradient calculation during evaluation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # Get the predicted class (the index of the max log-probability)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0) # Accumulate total number of samples
            correct += (predicted == target).sum().item() # Accumulate correct predictions

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f}%')
    return accuracy

# --- Main Execution Block ---
if __name__ == '__main__':
    # Check if a GPU is available and use it, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model and move it to the selected device
    model = DigitClassifier().to(device)

    # Define the loss function (Cross-Entropy Loss is common for classification)
    criterion = nn.CrossEntropyLoss()
    # Define the optimizer (Adam is a popular choice)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Load data
    train_loader, test_loader = load_mnist_data()

    # Train the model
    train_model(model, train_loader, criterion, optimizer, device, epochs=5)

    # Evaluate the model
    evaluate_model(model, test_loader, device)

    # Optional: Save the trained model
    torch.save(model.state_dict(), 'mnist_classifier.pth')
    print("Model saved to mnist_classifier.pth")