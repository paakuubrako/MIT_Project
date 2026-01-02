import torch
from torchvision import transforms
from PIL import Image
from src.cnn.cnn import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN()
model.load_state_dict(torch.load("cnn_model.pt", map_location=device))
model.eval()
model.to(device)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()

    return "Authentic" if pred == 0 else "Tampered"


if __name__ == "__main__":
    print(predict("example_patch.png"))
