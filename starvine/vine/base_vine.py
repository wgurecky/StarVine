##
# \brief Base vine class


class BaseVine(object):
    """!
    @brief Regular vine base class.
    """
    def __init__(self, data, weights=None):
        self.nT = data.shape[0]
        self.vS = np.zeros((self.nT, self.nT))

    def loadVineStructure(self, vS):
        """!
        @brief Load saved vine structure
        """
        pass
