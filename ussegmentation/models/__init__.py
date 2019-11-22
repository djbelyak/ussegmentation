from ussegmentation.models.empty import EmptyNet
from ussegmentation.models.enet import ENet


def get_model_list():
    return [EmptyNet, ENet]


def get_model_by_name(model_name):
    for model in get_model_list():
        if model.name == model_name:
            return model()
