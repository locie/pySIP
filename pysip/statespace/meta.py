from .nodes import Node
from ..utils import Namespace

model_registry = Namespace()


def statespace(cls, *args, **kwargs):
    if cls not in model_registry:
        return None
    if args or kwargs:
        return model_registry[cls](*args, **kwargs)
    return model_registry[cls]


class MetaStateSpace(type):
    def __new__(cls, name, bases, attrs):
        new_class = super(MetaStateSpace, cls).__new__(cls, name, bases, attrs)
        if "__base__" not in attrs:
            model_registry[attrs.get("__name__") or attrs["__qualname__"]] = new_class
        return new_class

    def __init__(self, name, bases, attr):
        if not self.__doc__:
            self.__doc__ = ""

        lines = []
        sections = [("inputs", "Inputs"), ("outputs", "Outputs"), ("states", "States")]

        for attr, title in sections:
            if hasattr(self, attr):
                lines += ["", f"{title}"]
                for x in getattr(self, attr):
                    x = Node(*x)
                    lines.append(
                        f"\t* ``{x.name}``: {x.description} `({x.category.value[1]})`"
                    )

        if hasattr(self, "params"):
            lines += ["", "**Model parameters**", ""]
            categories = set([Node(*x).category for x in self.params])
            categories = {c: [] for c in categories}
            for x in self.params:
                x = Node(*x)
                categories[x.category].append(x)
            for c in categories:
                lines += ["", f"\t* {c.value[0]}"]
                for x in categories[c]:
                    lines.append(
                        f"\t\t* ``{x.name}``: {x.description} `({x.category.value[1]})`"
                    )
                lines += [""]

        self.__doc__ = "\n".join([self.__doc__] + lines)
