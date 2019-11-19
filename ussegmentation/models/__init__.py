from ussegmentation.models.empty import EmptyNet


def get_model_list():
    return [EmptyNet]


def get_model_by_name(model_name):
    for model in get_model_list():
        if model.name == model_name:
            return model()
