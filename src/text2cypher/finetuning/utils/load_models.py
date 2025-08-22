import importlib
from loguru import logger

MODEL_CLASSES = {
    "t5": "text2cypher.finetuning.models.t5_model.T5NoteGenerationModel",
    # Add more model types here as needed
}


def load_class_from_path(class_path):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def load_model(model_ckpt, model_name, model_type, device, peft_method=None):
    if model_type not in MODEL_CLASSES:
        raise ValueError(f"Unsupported model type: {model_type}")

    logger.info(f"Loading model `{model_name}` from `{model_ckpt}` using type `{model_type}`")

    model_class = load_class_from_path(MODEL_CLASSES[model_type])

    model = model_class.load_model_from_checkpoint(
        checkpoint_path=model_ckpt,
        model_name=model_name,
        model_type=model_type,
        peft_method=peft_method
    )

    model.to(device)
    return model
