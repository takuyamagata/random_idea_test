'''
Test how to hand over initialisation arguments to sub modules with *args and **kwargs
'''


class Sub:
    def __init__(self, a=0, b=0, *args, **kwargs):
        self.a = a
        self.b = b
        return

    def print(self):
        print(f'a={self.a}, b={self.b}')
        return

class Top:
    def __init__(self, x, y=0, *args, **kwargs):
        self.sub = Sub(**kwargs)
        self.x = x
        self.y = y
        return

    def print(self):
        print(f'x={self.x}, y={self.y}')
        self.sub.print()
        return

if __name__ == '__main__':
    top = Top(10, 20, a=10, b=20, c=400)
    top.print()
