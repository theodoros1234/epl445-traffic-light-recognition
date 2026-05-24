import ultralytics
ultralytics.checks()

dataset_path ="path_to_dataset"
project_name = "project_name"
trained_model_name = "trained_model_name"

model = ultralytics.YOLO("yolo11n.pt")   # Pre-trained

# To freeze backbone for first 50 epochs.
model.train(
    data=dataset_path,    
    batch=16,
    save_period=10,
    device=0,
    save=True,
    cache=False,
    exist_ok=True,
    project=project_name,
    name=trained_model_name,
    pretrained=True,

    # Importart hyperparameters:
    epochs=50,     # unfreeze after 50 epochs and train until 250 epochs
    imgsz=640,
    freeze=11,     # 11 layers for YOLO backbone 
)

# Resume training with unfrozen layers.
# model = ultralytics.YOLO("runs/detect/" + project_name + "/freeze_11/weights/last.pt") # To continue training with all the layers unfrozen.

# model.train(
#     data=dataset_path,   
#     batch=16,
#     save_period=10,
#     device=0,
#     save=True,
#     cache=False,
#     exist_ok=True,
#     project=project_name,
#     name=trained_model_name,
#     pretrained=True,

#     # Importart hyperparameters:    
#     epochs=200,        # Last 200 epochs without frozen layers.   
#     imgsz=640, 
#     freeze=0
# )

# To resume later:
# model = ultralytics.YOLO("runs/detect/" + project_name + "/" + trained_model_name + "/weights/last.pt")
# model.train(resume=True, epochs = x) # x is total epochs when counting the previous epochs as well