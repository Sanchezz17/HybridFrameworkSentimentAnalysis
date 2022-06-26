from random import random
from typing import List


def initialize_population(population_size: int, chromosome_length: int) -> List[List[bool]]:
    return [[random() < 0.5 for _ in range(chromosome_length)] for _ in range(population_size)]
