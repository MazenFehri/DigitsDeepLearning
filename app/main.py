import io, base64
import numpy as np
from PIL import Image
import torch

def preprocess(image_b64: str):
    _, encoded = image_b64.split(",", 1)          # strip "data:image/png;base64,"
    img = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("L")  # grayscale
    img = img.resize((28, 28))
    arr = np.array(img, dtype=np.float32) / 255.0        # match ToTensor()'s [0,1] scaling
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # -> [1, 1, 28, 28]
    return tensor, img


def image_to_b64(img: Image.Image) -> str:
    # ponytail: debug-only, upscaled with NEAREST so the actual 28x28 pixels stay visible
    debug_img = img.resize((140, 140), Image.NEAREST)
    buf = io.BytesIO()
    debug_img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


from fastapi import FastAPI
from pydantic import BaseModel
from model import MNIST_v2, MNIST_v3

app = FastAPI()


def try_load(model_cls, path):
    # ponytail: v3 may not exist yet if training hasn't finished/been run
    try:
        m = model_cls(input_shape=1, hidden_units=120, output_shape=10)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()
        return m
    except FileNotFoundError:
        return None


models = {
    "v2": try_load(MNIST_v2, "model/mnist_v2.pth"),
    "v3": try_load(MNIST_v3, "model/mnist_v3.pth"),
}


def run_model(model, x):
    if model is None:
        return None
    with torch.inference_mode():
        probs = torch.softmax(model(x), dim=1)
        digit = int(probs.argmax(dim=1))
        return {"digit": digit, "confidence": float(probs[0, digit])}


class PredictRequest(BaseModel):
    image: str   # base64 data URL from canvas.toDataURL()

@app.post("/predict")
def predict(req: PredictRequest):
    x, resized_img = preprocess(req.image)
    return {
        "predictions": {name: run_model(m, x) for name, m in models.items()},
        "debug_image": image_to_b64(resized_img),
    }


from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")