from sklearn.base import ClassifierMixin

def get_model_from_config(
    model_package: str, model_class: str
) -> ClassifierMixin:
    if model_package == "sklearn.ensemble":
        import sklearn.ensemble as package

        model_class = getattr(package, model_class)
    elif model_class == "sklearn.tree":
        import sklearn.tree as package

        model_class = getattr(package, model_class)
    else:
        raise ValueError(f"Unsuppoted model package: {model_class}")

    return model_class

