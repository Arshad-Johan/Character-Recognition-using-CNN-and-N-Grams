import json
import torch
from flask import Flask, render_template, request
from PIL import Image, ImageChops, ImageOps
from torchvision import transforms

from model import Model
from train import SAVE_MODEL_PATH

app = Flask(__name__)
predictor = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/DigitRecognition", methods=["POST"])
def predict_digit():
    img = Image.open(request.files["img"]).convert("L")
    res_json = {"pred": "", "probs": [0] * 36}
    if predictor is not None:
        res = predictor(img)
        pred_index = int(res.argmax())
        res_json["pred"] = index_to_label(pred_index)
        res_json["probs"] = (res * 100).tolist()
    return json.dumps(res_json)

def index_to_label(index):
    if 0 <= index <= 9:
        return str(index)
    elif 10 <= index <= 35:
        return chr(index - 10 + ord('A'))
    return ""

class Predict():
    def __init__(self):
        device = torch.device("cpu")
        self.model = Model(num_classes=36).to(device)
        self.model.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=device))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def _centering_img(self, img):
        left, top, right, bottom = img.getbbox()
        w, h = img.size[:2]
        shift_x = (left + (right - left) // 2) - w // 2
        shift_y = (top + (bottom - top) // 2) - h // 2
        return ImageChops.offset(img, -shift_x, -shift_y)

    def __call__(self, img):
        img = ImageOps.invert(img)
        img = self._centering_img(img)
        img = img.resize((28, 28), Image.BICUBIC)
        tensor = self.transform(img)
        tensor = tensor.unsqueeze_(0)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(tensor)
            preds = preds.detach().numpy()[0]
        return preds

if __name__ == "__main__":
    import os
    assert os.path.exists(SAVE_MODEL_PATH), "no saved model"
    predictor = Predict()
    app.run(host="0.0.0.0", port=5001)
