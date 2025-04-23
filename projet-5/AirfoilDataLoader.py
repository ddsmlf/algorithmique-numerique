import sys
import numpy as np
import re  # regexp
import matplotlib.pyplot as plt

class AirfoilDataLoader:
    """
    A class to load airfoil data from a file.
    """

    def __init__(self, file_path):
        """
        Initialize the AirfoilDataLoader with a file path.

        Args:
            file_path (str): The path to the airfoil data file.
        """
        self.file_path = file_path
        self.dimension, self.outer_x, self.outer_y, self.inner_x, self.inner_y = self.load(file_path)

    @staticmethod
    def load(file_path):
        """
        Load the airfoil data from a file.

        Args:
            file_path (str): The path to the airfoil data file.

        Returns:
            tuple: A tuple containing the dimension, outer x-coordinates, outer y-coordinates, inner x-coordinates, and inner y-coordinates.
        """
        with open(file_path, 'r', encoding='UTF-8') as file:
            match_line = lambda line: re.match(r"\s*([\d\.-]+)\s*([\d\.-]+)", line)
            outer_coordinates = []; inner_coordinates = []
            reading_header = False; reading_outer = False; reading_inner = False
            for line in file:
                match = match_line(line)
                if match is None:
                    if not reading_header:
                        reading_header = True
                    elif not reading_outer:
                        reading_outer = True
                    elif not reading_inner:
                        reading_inner = True
                    continue
                if match is not None and reading_header and not reading_outer and not reading_inner:
                    dimension = np.array(list(map(lambda t: float(t), match.groups())))
                    continue
                if match is not None and reading_header and reading_outer and not reading_inner:
                    outer_coordinates.append(match.groups())
                    continue
                if match is not None and reading_header and reading_outer and reading_inner:
                    inner_coordinates.append(match.groups())
                    continue
            outer_x = np.array(list(map(lambda t: float(t[0]), outer_coordinates)))
            outer_y = np.array(list(map(lambda t: float(t[1]), outer_coordinates)))
            inner_x = np.array(list(map(lambda t: float(t[0]), inner_coordinates)))
            inner_y = np.array(list(map(lambda t: float(t[1]), inner_coordinates)))
            return dimension, outer_x, outer_y, inner_x, inner_y

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Please provide an input file")
        exit(1)

    loader = AirfoilDataLoader(sys.argv[1])
    print(loader.dimension)
    print(loader.outer_x)
    print(loader.outer_y)
    print(loader.inner_x)
    print(loader.inner_y)