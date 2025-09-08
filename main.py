import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
import os

#model = models.vgg19(pretrained=True).features
#print(model) # 0, 5, 10, 19, 28

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)
image_size = 256

loader = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = ["0", "5", "10", "19", "28"]
        self.model = models.vgg19(pretrained=True).features[:29]
    
    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features


def load_image(image_name):
    image = Image.open(image_name)
    try:
        exif = image._getexif()
        if exif:
            orientation = exif.get(0x0112)
            if orientation:
                if orientation == 3:
                    image = image.rotate(180, expand=True)
                elif orientation == 6:
                    image = image.rotate(270, expand=True)
                elif orientation == 8:
                    image = image.rotate(90, expand=True)
    except:
        pass  # If EXIF data doesn't exist or can't be read
    image = loader(image).unsqueeze(0)
    return image.to(device)


def run_model(image_name, style_name):
    original_img = load_image(f"Images/{image_name}.jpg")
    style_img = load_image(f"Styles/{style_name}.jpg")

    model = VGG().to(device).eval()
    generated = original_img.clone().requires_grad_(True)

    # Hyperparameters
    total_steps = 6000
    learning_rate = 0.001
    alpha = 1
    beta = 0.01
    optimizer = optim.Adam([generated], lr=learning_rate)

    for step in range(total_steps):
        #print(f"Step: {step}")
        generated_features = model(generated)
        original_features = model(original_img)
        style_features = model(style_img)

        original_loss = 0
        style_loss = 0
        for generated_feature, original_feature, style_feature in zip(generated_features, original_features, style_features):
            batch_size, channel, height, width = generated_feature.shape
            original_loss += torch.mean((generated_feature- original_feature)**2)

            # Gram matrix
            G = generated_feature.view(channel, height*width).mm(
                generated_feature.view(channel, height*width).t()
            )
            S = style_feature.view(channel, height*width).mm(
                style_feature.view(channel, height*width).t()
            )

            style_loss += torch.mean((G-S)**2)
        total_loss = alpha*original_loss + beta*style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if step%200 == 0:
            print(total_loss)
            save_image(generated, f"GeneratedImages/generated_{image_name}.png")


def main():
    while True:
        image_name = input("Enter the image name: ")
        try:
            image = Image.open("Images/" + image_name + ".jpg")
            break
        except:
            print("Error in finding image file")
    while True:
        style_name = input("Enter the style name: ")
        try:
            style = Image.open("Styles/" + style_name + ".jpg")
            break
        except:
            print("Error in finding style file")
    run_model(image_name, style_name)

if __name__ == "__main__":
    main()
