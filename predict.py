import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import glob
from src.cnn import cnn
from src.cifar_loader import extract_cifar_image

class ModelPredictor:
    def __init__(self):
        self.model = None
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        # CIFAR-10 normalization values
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    def load_model(self, model_path):
        """Load a .pth model file"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found!")

        print(f"Loading model from {model_path}...")
        self.model = cnn()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")

    def predict_image(self, image_path):
        """Make a prediction on a single image"""
        if self.model is None:
            raise ValueError("No model loaded! Please load a model first.")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file '{image_path}' not found!")

        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        return {
            'predicted_class': self.classes[predicted_class],
            'predicted_index': predicted_class,
            'confidence': confidence,
            'all_probabilities': {self.classes[i]: prob.item() for i, prob in enumerate(probabilities[0])}
        }

def list_available_models():
    """List all available .pth model files"""
    model_patterns = ['models/**/*.pth', '*.pth']
    models = []

    for pattern in model_patterns:
        models.extend(glob.glob(pattern, recursive=True))

    return sorted(set(models))

def list_image_files():
    """List common image files in current directory"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    images = []

    for ext in image_extensions:
        images.extend(glob.glob(ext))
        images.extend(glob.glob(ext.upper()))

    return sorted(images)

def main():
    predictor = ModelPredictor()

    print("=== CIFAR-10 Model Predictor ===\n")

    # Model selection
    available_models = list_available_models()

    if not available_models:
        print("No .pth model files found!")
        print("Please place a .pth model file in this directory or the models/ folder.")
        return

    print("Available models:")
    for i, model in enumerate(available_models, 1):
        print(f"{i}. {model}")

    while True:
        try:
            choice = input(f"\nSelect a model (1-{len(available_models)}) or enter path: ").strip()

            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(available_models):
                    model_path = available_models[idx]
                    break
                else:
                    print("Invalid selection!")
            elif os.path.exists(choice) and choice.endswith('.pth'):
                model_path = choice
                break
            else:
                print("Invalid model path or file doesn't exist!")
        except (ValueError, KeyboardInterrupt):
            print("\nExiting...")
            return

    # Load the selected model
    try:
        predictor.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Image prediction loop
    while True:
        print(f"\n{'='*50}")
        print("Image Prediction")
        print(f"{'='*50}")

        # Show available images
        available_images = list_image_files()
        if available_images:
            print("\nAvailable images in current directory:")
            for i, img in enumerate(available_images, 1):
                print(f"{i}. {img}")

        print("\nOptions:")
        print("- Enter image path")
        print("- Type 'cifar' to extract a random CIFAR-10 test image")
        print("- Type 'cifar:train:123' to extract specific image (index 123 from train set)")
        print("- Type 'q' to quit")

        image_input = input("\nYour choice: ").strip()

        if image_input.lower() == 'q':
            break

        # Handle CIFAR image extraction
        if image_input.lower().startswith('cifar'):
            try:
                parts = image_input.split(':')
                if len(parts) == 1:
                    # Random test image
                    image_path, true_class, true_index = extract_cifar_image()
                    print(f"Extracted random CIFAR-10 test image: {true_class} (class {true_index})")
                elif len(parts) == 3:
                    # Specific image: cifar:train:123
                    dataset = parts[1]
                    index = int(parts[2])
                    filename = f"cifar_{dataset}_{index}.png"
                    image_path, true_class, true_index = extract_cifar_image(
                        index=index, dataset=dataset, save_path=filename
                    )
                    print(f"Extracted CIFAR-10 {dataset} image #{index}: {true_class} (class {true_index})")
                else:
                    print("Invalid CIFAR format! Use 'cifar' or 'cifar:train:123'")
                    continue
            except (ValueError, IndexError) as e:
                print(f"Error extracting CIFAR image: {e}")
                continue
        # Handle numbered selection
        elif image_input.isdigit() and available_images:
            idx = int(image_input) - 1
            if 0 <= idx < len(available_images):
                image_path = available_images[idx]
            else:
                print("Invalid selection!")
                continue
        else:
            image_path = image_input

        # Make prediction
        try:
            result = predictor.predict_image(image_path)

            print(f"\n📸 Image: {image_path}")
            print(f"🎯 Prediction: {result['predicted_class']}")
            print(f"📊 Confidence: {result['confidence']:.2%}")

            print(f"\n📈 All class probabilities:")
            for class_name, prob in result['all_probabilities'].items():
                bar_length = int(prob * 20)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"  {class_name:8} {bar} {prob:.2%}")

        except Exception as e:
            print(f"Error making prediction: {e}")

if __name__ == "__main__":
    main()