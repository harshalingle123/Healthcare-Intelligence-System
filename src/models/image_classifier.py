"""
Medical Image Classifier using DenseNet-121 with Grad-CAM explainability.
Part of the Healthcare Intelligence System models layer.
"""

import os
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torchvision import models, transforms
    from PIL import Image
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        roc_auc_score,
        precision_recall_fscore_support,
    )

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class MedicalImageClassifier:
    """Chest X-ray / medical image classification with DenseNet-121 and Grad-CAM."""

    def __init__(self, num_classes=5, image_size=224):
        self.num_classes = num_classes
        self.image_size = image_size
        self.model = None
        self.device = None
        self.class_names = None
        self.training_history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": []
        }

        if not TORCH_AVAILABLE:
            print("WARNING: PyTorch/torchvision not installed. "
                  "Install with: pip install torch torchvision Pillow scikit-learn")

    class ChestXrayDataset(Dataset):
        """PyTorch Dataset for loading chest X-ray images."""

        def __init__(self, image_paths, labels=None, transform=None):
            self.image_paths = image_paths
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            if self.labels is not None:
                label = self.labels[idx]
                return image, label
            return image

    def build_model(self):
        """Load DenseNet-121 pretrained, replace classifier for num_classes."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and torchvision are required.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = models.densenet121(pretrained=True)

        # Freeze early layers
        for param in self.model.features.parameters():
            param.requires_grad = False
        # Unfreeze last dense block for fine-tuning
        for param in self.model.features.denseblock4.parameters():
            param.requires_grad = True

        # Replace classifier: DenseNet-121 has 1024 features before classifier
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, self.num_classes)

        self.model.to(self.device)
        return self.model

    def get_transforms(self, train=True):
        """Return image transforms for training or validation."""
        if train:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.Resize(self.image_size + 32),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.image_size + 32),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

    def train(self, train_loader, val_loader, epochs=10, lr=1e-4):
        """
        Training loop with CrossEntropyLoss (class weights), AdamW, ReduceLROnPlateau,
        early stopping, and mixed precision.
        """
        if self.model is None:
            self.build_model()

        # Compute class weights from training data
        all_labels = []
        for _, labels in train_loader:
            all_labels.extend(labels.numpy())
        all_labels = np.array(all_labels)
        class_counts = np.bincount(all_labels, minlength=self.num_classes)
        total = len(all_labels)
        class_weights = total / (self.num_classes * class_counts.astype(float) + 1e-6)
        class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=1e-4
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        best_val_loss = float("inf")
        patience = 3
        patience_counter = 0

        self.training_history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": []
        }

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct = 0
            total_samples = 0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

            train_loss = running_loss / total_samples
            train_acc = correct / total_samples
            self.training_history["train_loss"].append(train_loss)
            self.training_history["train_acc"].append(train_acc)

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        outputs = self.model(images)
                        loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)

            val_loss = val_loss / val_total
            val_acc = val_correct / val_total
            self.training_history["val_loss"].append(val_loss)
            self.training_history["val_acc"].append(val_acc)

            scheduler.step(val_loss)

            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best weights
                self._best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    self.model.load_state_dict(self._best_state)
                    break

        return self.training_history

    def predict(self, image_path):
        """Single image prediction returning (class_label, probabilities)."""
        if self.model is None:
            raise RuntimeError("Model has not been built/trained.")

        self.model.eval()
        transform = self.get_transforms(train=False)

        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1).cpu().numpy()[0]

        predicted_class = int(np.argmax(probabilities))

        if self.class_names:
            class_label = self.class_names[predicted_class]
        else:
            class_label = predicted_class

        return class_label, probabilities

    def generate_gradcam(self, image_path, target_class=None):
        """
        Grad-CAM visualization: hook into last conv layer, compute
        gradient-weighted activation map. Returns heatmap as numpy array.
        """
        if self.model is None:
            raise RuntimeError("Model has not been built/trained.")

        self.model.eval()

        # Hook into the last conv layer (features.denseblock4)
        activations = []
        gradients = []

        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        # Register hooks on the last batch norm in features
        target_layer = self.model.features.norm5
        fwd_handle = target_layer.register_forward_hook(forward_hook)
        bwd_handle = target_layer.register_full_backward_hook(backward_hook)

        # Forward pass
        transform = self.get_transforms(train=False)
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(self.device)
        input_tensor.requires_grad_(True)

        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass for the target class
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        # Compute Grad-CAM
        act = activations[0].detach()  # (1, C, H, W)
        grad = gradients[0].detach()   # (1, C, H, W)

        # Global average pooling of gradients
        weights = grad.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * act).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to original image size
        from PIL import Image as PILImage
        cam_resized = np.array(
            PILImage.fromarray((cam * 255).astype(np.uint8)).resize(
                (self.image_size, self.image_size), PILImage.BILINEAR
            )
        ) / 255.0

        # Clean up hooks
        fwd_handle.remove()
        bwd_handle.remove()

        return cam_resized

    def evaluate(self, test_loader):
        """Compute per-class precision/recall/F1, confusion matrix, AUC-ROC."""
        if self.model is None:
            raise RuntimeError("Model has not been built/trained.")

        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = self.model(images)
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                all_probs.append(probs)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.concatenate(all_probs, axis=0)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None
        )

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        # AUC-ROC (one-vs-rest)
        try:
            if self.num_classes == 2:
                auc = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                auc = roc_auc_score(
                    all_labels, all_probs, multi_class="ovr", average="weighted"
                )
        except ValueError:
            auc = None

        report = classification_report(
            all_labels, all_preds,
            target_names=self.class_names if self.class_names else None,
            output_dict=True
        )

        results = {
            "per_class_precision": precision.tolist(),
            "per_class_recall": recall.tolist(),
            "per_class_f1": f1.tolist(),
            "confusion_matrix": cm.tolist(),
            "auc_roc": auc,
            "classification_report": report,
        }
        return results

    def save_model(self, path):
        """Save model state dict."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "num_classes": self.num_classes,
            "image_size": self.image_size,
            "class_names": self.class_names,
        }, path)

    def load_model(self, path):
        """Load model state dict."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=self.device)

        self.num_classes = checkpoint.get("num_classes", self.num_classes)
        self.image_size = checkpoint.get("image_size", self.image_size)
        self.class_names = checkpoint.get("class_names", None)

        self.build_model()
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        return self.model
