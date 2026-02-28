from roboflow import Roboflow

rf = Roboflow(api_key="GwvyBMdopkkBWJ39oH0V")  

project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
dataset = project.version(4).download("yolov8")

print("Dataset downloaded!")
print("Dataset location:", dataset.location)