"""
Wine Recommender Neural Network
Uses PyTorch to create embeddings for wine recommendations
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt


class WineRecommenderNN(nn.Module):
    """
    Neural Network for Wine Recommendation
    Creates low-dimensional embeddings from high-dimensional wine features
    """

    def __init__(self, input_dim, hidden_dim=512, output_dim=64):
        """
        Initialize the neural network

        Args:
            input_dim (int): Dimension of input features (TF-IDF vector size)
            hidden_dim (int): Dimension of hidden layer
            output_dim (int): Dimension of output embeddings
        """
        super(WineRecommenderNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Define network architecture
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim * 3 // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 3 // 4),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 3 // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        # Decoder for autoencoder training
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim * 3 // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 3 // 4),
            nn.Linear(hidden_dim * 3 // 4, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.to(self.device)

    def forward(self, x):
        """
        Forward pass through the autoencoder

        Args:
            x (torch.Tensor): Input features

        Returns:
            tuple: (embeddings, reconstructed)
        """
        embeddings = self.encoder(x)
        reconstructed = self.decoder(embeddings)
        return embeddings, reconstructed

    def get_embeddings(self, features):
        """
        Get low-dimensional embeddings for wine features

        Args:
            features (np.ndarray or sparse matrix): Wine feature vectors

        Returns:
            np.ndarray: Low-dimensional embeddings
        """
        self.eval()

        # Convert sparse matrix to dense if needed
        if hasattr(features, "toarray"):
            features = features.toarray()

        # Convert to tensor
        features_tensor = torch.FloatTensor(features).to(self.device)

        with torch.no_grad():
            embeddings, _ = self.forward(features_tensor)
            embeddings = embeddings.cpu().numpy()

        return embeddings

    def train_model(
        self, features, epochs=50, batch_size=128, learning_rate=0.001, verbose=True
    ):
        """
        Train the neural network using autoencoder approach

        Args:
            features (np.ndarray or sparse matrix): Wine feature vectors
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for optimizer
            verbose (bool): Whether to print training progress
        """
        self.train()

        # Convert sparse matrix to dense if needed
        if hasattr(features, "toarray"):
            features = features.toarray()

        # Convert to tensor
        features_tensor = torch.FloatTensor(features)

        # Create DataLoader
        dataset = TensorDataset(features_tensor, features_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Define optimizer and loss
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        # Track loss history
        loss_history = []

        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                embeddings, reconstructed = self.forward(batch_x)

                # Calculate reconstruction loss
                loss = criterion(reconstructed, batch_y)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            loss_history.append(avg_loss)
            scheduler.step(avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

        if verbose:
            print("Training completed!")

        return loss_history

    def save_model(self, filepath):
        """
        Save the trained model to disk

        Args:
            filepath (str): Path to save the model
        """
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save model state and configuration
        model_data = {
            "state_dict": self.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """
        Load a trained model from disk

        Args:
            filepath (str): Path to the saved model
        """
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        # Verify dimensions match
        if (
            model_data["input_dim"] != self.input_dim
            or model_data["hidden_dim"] != self.hidden_dim
            or model_data["output_dim"] != self.output_dim
        ):
            raise ValueError(
                f"Model dimensions mismatch. Expected ({self.input_dim}, "
                f"{self.hidden_dim}, {self.output_dim}), got "
                f"({model_data['input_dim']}, {model_data['hidden_dim']}, "
                f"{model_data['output_dim']})"
            )

        # Load state dict
        self.load_state_dict(model_data["state_dict"])
        self.eval()

        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Train the neural network on real wine data
    print("Training Wine Recommender Neural Network on real data...")
    print("=" * 60)

    HIDDEN_DIM = 512
    OUTPUT_DIM = 64
    EPOCHS = 500
    BATCH_SIZE = 256
    LEARNING_RATE = 0.0005

    # Load the real data
    data_dir = Path(__file__).parent / "data"
    plot_dir = Path(__file__).parent / "plots"
    plot_dir.mkdir(exist_ok=True)

    print("\nLoading feature vectors...")
    with open(data_dir / "feature_vectors.pkl", "rb") as f:
        feature_vectors = pickle.load(f)

    print(f"Loaded feature vectors: {feature_vectors.shape}")
    print(f"Number of wines: {feature_vectors.shape[0]}")
    print(f"Feature dimension: {feature_vectors.shape[1]}")

    # Initialize model with actual dimensions
    input_dim = feature_vectors.shape[1]
    model = WineRecommenderNN(
        input_dim=input_dim, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM
    )

    # Create architecture-specific subdirectory name
    arch_name = f"{HIDDEN_DIM}-{HIDDEN_DIM * 3 // 4}-{HIDDEN_DIM // 2}-{OUTPUT_DIM}"
    print(f"Architecture: {arch_name}")

    # Train model on real data
    print("\n" + "=" * 60)
    print("Training on real wine TF-IDF features...")
    print("=" * 60)
    loss_history = model.train_model(
        feature_vectors,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        verbose=True,
    )

    # Plot training loss
    print("\n" + "=" * 60)
    print("Plotting training loss...")
    plt.figure(figsize=(10, 6))
    # Omit the first value to avoid large initial loss skewing the plot
    plt.plot(range(2, len(loss_history) + 1), loss_history[1:], "b-", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (MSE)", fontsize=12)
    plt.title("Training Loss Over Time", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot in architecture-specific subdirectory with learning rate folder
    lr_folder = f"lr{str(LEARNING_RATE).replace('0.', '')}"
    arch_plot_dir = plot_dir / lr_folder / arch_name
    arch_plot_dir.mkdir(exist_ok=True, parents=True)
    plot_path = arch_plot_dir / "training_loss.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Loss plot saved to: {plot_path}")
    plt.close()

    # Save trained model in architecture-specific subdirectory
    print("\n" + "=" * 60)
    trained_dir = Path(__file__).parent / "trained" / arch_name
    trained_dir.mkdir(exist_ok=True, parents=True)
    model_path = trained_dir / "wine_nn_model.pkl"
    model.save_model(str(model_path))

    embeddings = model.get_embeddings(feature_vectors)

    # Test load functionality
    print("\n" + "=" * 60)
    print("Testing model load...")
    model2 = WineRecommenderNN(
        input_dim=input_dim, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM
    )
    model2.load_model(str(model_path))

    # Verify embeddings are consistent
    embeddings2 = model2.get_embeddings(feature_vectors[:100])
    print(f"Embeddings match after load: {np.allclose(embeddings[:100], embeddings2)}")
