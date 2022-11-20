import os
import pickle
import warnings

from pysip import __version__


def with_extension(extension):
    def _extension(f):
        def wrapper(*args):
            a = args[0]
            if not a.endswith("." + extension):
                a += "." + extension
            return f(a, *args[1:])

        return wrapper

    return _extension


@with_extension("pickle")
def save_model(path, model):
    model.__version__ = __version__
    try:
        pass
        model.ss.delete_continuous_dssm()
    except Exception:
        warnings.warn(
            "Deleting continuous dssm while serializing model hasn't been successfull."
        )
    with open(path, "wb") as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


@with_extension("pickle")
def load_model(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    try:
        model.ss.init_continuous_dssm()
        model.ss.update_continuous_dssm()
    except Exception:
        warnings.warn(
            "Updating continuous dssm while deserializing model hasn't been successfull."
        )
    return model
