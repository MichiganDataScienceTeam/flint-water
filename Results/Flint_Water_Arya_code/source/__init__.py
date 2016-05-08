import warnings

try: ImportWarning
except NameError:
    class ImportWarning(Warning):
        pass

from mainPipeline import mainPipeline

