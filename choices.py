from enum import Enum


class Choices(str, Enum):
    @classmethod
    def choices(cls):
        return [choice.value for choice in cls]


class ModelType(Choices):
    CLASSIFICATION = "Classification"
    REGRESSION = "Regression"
    SURVIVAL = "Survival Analysis"


class EvaluationType(Choices):
    ONLY_TRAINING = "Only training"
    TRAIN_TEST = "Training and testing - Hold out"
    CROSS_VALIDATION = "Cross Validation"


class DivType(Choices):
    BY_ORDER = "By order in dataset"
    RANDOM = "Random"
    STRATIFIED = "Stratified"
