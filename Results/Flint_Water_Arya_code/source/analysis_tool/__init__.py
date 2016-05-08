import warnings

try: ImportWarning
except NameError:
    class ImportWarning(Warning):
        pass

from graph_analysis import analysis_n_neighbour_graph


