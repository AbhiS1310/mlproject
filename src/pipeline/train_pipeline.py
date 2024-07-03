from src.components.model_trainer import ModelTrainer
from src.components.data_transformation import DataTransformation
from src.components.data_ingestion import DataIngestion


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):

        obj = DataIngestion()
        obj.initate_data_ingestion()
        data_transformer = DataTransformation()
        train_arr,test_arr,_ = data_transformer.intiate_data_transformation()
        model_trainer = ModelTrainer()
        model_trainer.intiate_model_trainer(train_arr,test_arr)