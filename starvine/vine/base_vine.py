##
# \brief Base vine class


class BaseVine(object):
    """!
    @brief Regular vine base class.
    """
    def __init__(self, data=None, weights=None):
        pass

    def loadVineStructure(self, vS):
        """!
        @brief Load saved vine structure
        """
        pass

    def vineNLLH(self, vineParams=[None], **kwargs):
        """!
        @brief Compute the vine negative log likelyhood.  Used for
        simulatneous MLE estimation of PCC model parameters.
        Loops through all tree levels and sums all NLL.
        @param vineParams <b>np_array</b>  Flattened array of all copula parametrs in vine
        """
        if not any(vineParams):
            self._initVineParams()
        self.nLLH = 0.
        for lvl, tree in enumerate(self.vine):
            treeNLLH = tree.treeNLLH(vineParams[self.vineParamsMap[lvl]:
                                                self.vineParamsMap[lvl + 1]])
            self.nLLH += treeNLLH
        return self.nLLH

    def _initVineParams(self):
        self.vineParams = []
        self.vineParamsMap = [0]
        for lvl, tree in enumerate(self.vine):
            self.vineParams.append(tree._initTreeParamMap())
            self.vineParamsMap.append(self.vineParamsMap[lvl] + len(self.vineParams[lvl]))

    def sfitMLE(self, **kwargs):
        """!
        @brief Simulataneously estimate all copula paramters in the
        vine by MLE.  Uses SLSQP method by default.
        """
        self._initVineParams()
        params0 = np.array(self.vineParams).flatten()
        self.fittedParams = minimize(self.vineNLLH, params0, args=(),
                                     method=kwargs.pop("method", "SLSQP"),
                                     tol=kwargs.pop("tol", 1e-5))

    def treeHfun(self, level=0):
        """!
        @brief Operates on a tree, T_(i).
        The conditional distribution is evaluated
        at each edge in the tree providing univariate distributions that
        populate the dataFrame in the tree level T_(i+1)
        """
        pass

    def sample(self, n=1000):
        """!
        @brief Draws n samples from the vine.
        """
        pass
