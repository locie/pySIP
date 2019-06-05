import pickle


def with_extension(extension):
    def _extension(f):
        def wrapper(*args):
            a = args[0]
            if not a.endswith('.' + extension):
                a += '.' + extension
            return f(a, *args[1:])
        return wrapper
    return _extension


@with_extension('pickle')
def save_model(path, model):
    with open(path, 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


@with_extension('pickle')
def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
