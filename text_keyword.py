class TextKeyword:
    def __init__(self, token: str, stem: str, pos: str, count: int):
        self.token = token
        self.stem = stem
        self.pos = pos
        self.count = count

    def __str__(self):
        return f"({self.token}, {self.stem}, {self.pos}, count: {self.count})"

    def __repr__(self):
        return str(self)
