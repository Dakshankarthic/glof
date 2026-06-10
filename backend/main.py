from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

# Enable CORS for Netlify (Allows the frontend to communicate with this backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Monkey-patch YOLO to support our custom CBAM Attention architecture
import ultralytics.nn.modules
import ultralytics.nn.tasks
from custom_modules import CBAM
ultralytics.nn.modules.CBAM = CBAM
ultralytics.nn.tasks.CBAM = CBAM

# Load YOLO model (this will soon be our YOLO11+CBAM model)
model = YOLO("best.pt")

@app.get("/")
def read_root():
    return {"status": "GLOF Detection API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run inference (Filtering out low confidence guesses!)
    results = model(img, conf=0.45)
    
    # Render results on image
    res_plotted = results[0].plot()

    # Convert back to image
    _, encoded_img = cv2.imencode('.jpg', res_plotted)
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")
