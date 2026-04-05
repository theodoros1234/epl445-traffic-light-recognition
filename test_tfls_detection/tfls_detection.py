from ultralytics import YOLO
from pathlib import Path
import cv2
from matplotlib import pyplot as plt

# Load a pretrained YOLO model (adjust model type as needed)
model = YOLO("yolo11m.pt")  # n, s, m, l, x versions available
# print(model.names) # Shows all classes the model can detect, as dictionary.

input_dir = Path("./images")   # Path for the image to detect.
output_dir = Path("./results") # Result path.

for img_path in input_dir.iterdir():
    # Perform object detection for just traffic lights.
    results = model.predict(source=img_path, classes=[9])
    
    # Result path for result.
    result_img_path = output_dir / img_path.name  
    
    # Removes class labels and confidence levels.
    img = results[0].plot(labels=False, conf=False)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   
    plt.imsave(result_img_path, img)