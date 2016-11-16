"""

This module contains a class, Configuration, that holds the configuration parameters of the running program.

To set attributes:
    import Configuration
    config = Configuration()
    config.x = y

To get attributes:
    config.x

"""

https://gist.github.com/niccokunzmann/5262590
http://stackoverflow.com/questions/4295678/understanding-the-difference-between-getattr-and-getattribute


class Configuration:

    # Default variables.
    columnSeparator = '\t'
    columnsToIgnore = []
    exampleIDColumn = None
    examplesPerShard = 100
    exampleTestFraction = 0.15
    exampleValidateFraction = 0.15
    headerPresent = False
    isLogging = True

    def __init__(self, **kwargs):
        # Initialise any arguments supplied at creation.
        for i, j in kwargs.items():
            self.__dict__[i] = j
