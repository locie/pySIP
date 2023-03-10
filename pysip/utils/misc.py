

class Namespace(dict):
    """A namespace object that allows attributes to be accessed as dict keys.

    This class is a subclass of dict, so it can be used as a dict.  It
    also allows attributes to be accessed as dict keys.  For example::

        >>> ns = Namespace()
        >>> ns['foo'] = 'bar'
        >>> ns.foo
        'bar'
        >>> ns.foo = 'baz'
        >>> ns['foo']
        'baz'

    """

    def __init__(self, *args, **kwargs):
        super(Namespace, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __repr__(self):
        return 'Namespace(%s)' % super(Namespace, self).__repr__()

    def __str__(self):
        return 'Namespace(%s)' % super(Namespace, self).__str__()

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError(name)

    def __dir__(self):
        return list(self.keys())

    def copy(self):
        return Namespace(self)

    def update(self, *args, **kwargs):
        if len(args) > 1:
            raise TypeError('update expected at most 1 arguments, got %d' %
                            len(args))
        other = dict(*args, **kwargs)
        for key, value in other.items():
            self[key] = value

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def pop(self, key, *args):
        if len(args) > 1:
            raise TypeError('pop expected at most 2 arguments, got %d' %
                            (len(args) + 1))
        if key in self:
            value = self[key]
            del self[key]
            return value
        elif args:
            return args[0]
        else:
            raise KeyError(key)

    def popitem(self):
        try:
            key = next(iter(self))
        except StopIteration:
            raise KeyError('popitem(): dictionary is empty')
        value = self[key]
        del self[key]
        return key, value

    def clear(self):
        for key in list(self.keys()):
            del self[key]

    def __reduce__(self):
        items = [[k, self[k]] for k in self]
        inst_dict = vars(self).copy()
        inst_dict.pop('__dict__', None)
        inst_dict.pop('__weakref__', None)
        return (self.__class__, (items,), inst_dict)

    def __copy__(self):
        return self.__class__(self)

    def __deepcopy__(self, memo):
        import copy
        return self.__class__(copy.deepcopy(dict(self), memo))

    def __eq__(self, other):
        if isinstance(other, Namespace):
            return dict(self) == dict(other)
        return dict(self) == other

    def __ne__(self, other):
        return not self == other
