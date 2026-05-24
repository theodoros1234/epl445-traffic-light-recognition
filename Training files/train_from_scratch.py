import ultralytics
ultralytics.checks()

dataset_path ="path_to_dataset"
project_name = "project_name"
trained_model_name = "trained_model_name"

model = ultralytics.YOLO("yolo11n.yaml") # From scratch

model.train(
    data=dataset_path,    
    batch=16,
    save_period=10,
    device=0,
    pretrained=False,
    project=project_name,
    name=trained_model_name,   
    save=True,
    cache=False,
    exist_ok=True,

    # Important hyperparameters
    epochs=250,
    imgsz=1280,

    # Optional hyperparameters.
    #lr0=0.001, # default = 0.01
    #lrf=0.01,  # default = 0.01
    #optimizer="AdamW", # Good optimizer for training using the specific learning rates.

   # cos_lr = True # Adjusts learning rate following a cosine curve over epochs. Helps for managing learning rate.    
)

# To resume later:
# model = ultralytics.YOLO("runs/detect/" + project_name + "/" + trained_model_name + "/weights/last.pt")
# model.train(resume=True, epochs = x) # x is total epochs when counting the previous epochs as well