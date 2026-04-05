from dataclasses import dataclass

def my_func(crystal=None, **kwargs):
    return MyClass(a=crystal*2, b=crystal*3)

@dataclass(kw_only=True)
class MyClass:
    a: int
    b: int

    def __init__(self, crystal=None, **kwargs):
        if crystal is not None:
            res = my_func(crystal=crystal)
            for k, v in vars(res).items():
                setattr(self, k, v)
        else:
            for k, v in kwargs.items():
                setattr(self, k, v)

obj1 = MyClass(a=1, b=2)
print("obj1", obj1.a, obj1.b)

obj2 = MyClass(crystal=5)
print("obj2", obj2.a, obj2.b)
