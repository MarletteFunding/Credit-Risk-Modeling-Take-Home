from datetime import datetime
import folder_manager
import os
import yaml

config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
home_dir = config['HOME_DIRECTORY']

def create_submodel(model_name:str):
    submodel_path = home_dir + "/outputs/"
    author = "EJ"
    
    folder_manager.submodel_name = datetime.now().strftime("%d_%H_%M") + "_"+model_name
    
    folder_manager.output_path = submodel_path+folder_manager.submodel_name
    
    folder_manager.encoding_path = folder_manager.output_path+"/encodings/"
    
    folder_manager.feature_report_path = folder_manager.output_path+"/feature_report/"
    

    os.mkdir(folder_manager.output_path)
    os.mkdir(folder_manager.encoding_path)
    os.mkdir(folder_manager.feature_report_path)
    