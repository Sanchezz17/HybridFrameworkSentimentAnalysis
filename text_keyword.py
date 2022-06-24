class TextKeyword:
    def __init__(self, token, stem, pos):
        self.token = token
        self.stem = stem
        self.pos = pos

    def __str__(self):
        return f"({self.token}, {self.stem}, {self.pos})"