def clone_model(model):
    cls = type(model)
    return cls(**model.get_params())
