from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.enhance_trainer import EnhanceTrainer

if __name__ == '__main__':
    di = DataIngestion()
    trainpath, test_path = di.initiate_ingestion()

    transformer = DataTransformation()
    trainarr, testarr = transformer.initiate_transformation(trainpath, test_path)

    trainer = EnhanceTrainer()
    trainer.initiate_training(trainarr, testarr)