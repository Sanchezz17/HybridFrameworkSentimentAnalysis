class TextKeyword:
    def __init__(self, token: str, stem: str, pos: str):
        self.token = token
        self.stem = stem
        self.pos = pos

    def __str__(self):
        return f"({self.token}, {self.stem}, {self.pos})"

    def __repr__(self):
        return str(self)