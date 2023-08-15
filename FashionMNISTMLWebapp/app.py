from flask import Flask, render_template, request, session
import torch
from model import model, test_data
from PIL import Image
from torchvision.transforms import transforms


app = Flask(__name__)

@app.route("/", methods = ["GET"])
def home():
    return render_template('index.html')


@app.route("/", methods = ["POST"])
def predict():
    imagefile = request.files["imagefile"]
    image_path = "./uploaded_images/" + imagefile.filename
    imagefile.save(image_path)

    #image_to_upload = session.get(image_path)

    # Loading image
    image = Image.open(image_path)
    # print(image.format)
    # print(image.size)


    # Using torchvision.transforms to manipulate the shape of input shape
    transform1 = transforms.Grayscale(1)
    transform2 = transforms.PILToTensor()

    # Applying transformation
    img_tensor = transform1(image)
    tensor_image = transform2(img_tensor)
    tensor_image = tensor_image.type(torch.float)
    print(f"tensor_image shape:{tensor_image.shape}, tensor_image dtype: {tensor_image.dtype}")


    # Making prediciton
    model.eval()
    with torch.inference_mode():
        class_label= torch.argmax(torch.softmax(model(tensor_image.unsqueeze(dim = 0)), dim = 1), dim = 1).item()
    class_name = test_data.classes[class_label]

    return render_template("index.html", prediction =class_name)


if __name__ == '__main__':
   app.run(debug=True)