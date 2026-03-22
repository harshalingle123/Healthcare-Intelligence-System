"""
Clinical NLP Model using Bio_ClinicalBERT for medical text classification.
Part of the Healthcare Intelligence System models layer.
"""

import os
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import BertTokenizer, BertForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class ClinicalNLPModel:
    """Clinical text classification using Bio_ClinicalBERT with PyTorch."""

    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT", num_classes=5):
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = None
        self.tokenizer = None
        self.device = None
        self.max_length = 512
        self.training_history = {"train_loss": [], "val_loss": []}

        if not TORCH_AVAILABLE:
            print("WARNING: PyTorch is not installed. Install with: pip install torch")
        if not TRANSFORMERS_AVAILABLE:
            print("WARNING: transformers is not installed. Install with: pip install transformers")

    def build_model(self):
        """Load BertForSequenceClassification with the specified number of classes."""
        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "PyTorch and transformers are required. "
                "Install with: pip install torch transformers"
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = BertForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_classes
            )
        except Exception as e:
            print(f"Could not load {self.model_name}: {e}")
            print("Falling back to bert-base-uncased.")
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=self.num_classes
            )

        self.model.to(self.device)
        return self.model

    def _create_dataset(self, texts, labels=None):
        """Tokenize texts and create a TensorDataset."""
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        if labels is not None:
            label_tensor = torch.tensor(labels, dtype=torch.long)
            dataset = TensorDataset(input_ids, attention_mask, label_tensor)
        else:
            dataset = TensorDataset(input_ids, attention_mask)

        return dataset

    def train(self, texts, labels, epochs=3, batch_size=16, lr=2e-5):
        """Train the model with AdamW, linear warmup, and early stopping."""
        if self.model is None:
            self.build_model()

        dataset = self._create_dataset(texts, labels)

        # 80/20 train-val split
        val_size = max(1, int(0.2 * len(dataset)))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)

        total_steps = len(train_loader) * epochs
        warmup_steps = int(0.1 * total_steps)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
            )

        scheduler = LambdaLR(optimizer, lr_lambda)

        # Early stopping
        best_val_loss = float("inf")
        patience = 2
        patience_counter = 0

        self.training_history = {"train_loss": [], "val_loss": []}
        global_step = 0

        self.model.train()
        for epoch in range(epochs):
            total_train_loss = 0.0
            self.model.train()

            for batch in train_loader:
                input_ids, attention_mask, batch_labels = (
                    batch[0].to(self.device),
                    batch[1].to(self.device),
                    batch[2].to(self.device),
                )

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=batch_labels
                )
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                global_step += 1

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            self.training_history["train_loss"].append(avg_train_loss)

            # Validation
            self.model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids, attention_mask, batch_labels = (
                        batch[0].to(self.device),
                        batch[1].to(self.device),
                        batch[2].to(self.device),
                    )
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=batch_labels
                    )
                    total_val_loss += outputs.loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            self.training_history["val_loss"].append(avg_val_loss)

            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        return self.training_history

    def predict(self, texts):
        """Batch inference returning (predictions, probabilities) as numpy arrays."""
        if self.model is None:
            raise RuntimeError("Model has not been built/trained. Call build_model() or train() first.")

        self.model.eval()
        dataset = self._create_dataset(texts)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        all_preds = []
        all_probs = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        predictions = np.concatenate(all_preds, axis=0)
        probabilities = np.concatenate(all_probs, axis=0)
        return predictions, probabilities

    def extract_embeddings(self, texts):
        """Extract CLS token embeddings (768-dim) for downstream fusion."""
        if self.model is None:
            raise RuntimeError("Model has not been built/trained. Call build_model() or train() first.")

        self.model.eval()
        dataset = self._create_dataset(texts)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        all_embeddings = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)

                outputs = self.model.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                # CLS token is the first token
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                all_embeddings.append(cls_embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def save_model(self, path):
        """Save model state dict and tokenizer."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "model_state.pt"))
        self.tokenizer.save_pretrained(path)

    def load_model(self, path):
        """Load model state dict and tokenizer."""
        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            raise ImportError("PyTorch and transformers are required.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes
        )
        state_dict = torch.load(
            os.path.join(path, "model_state.pt"),
            map_location=self.device
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        return self.model
