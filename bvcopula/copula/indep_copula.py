##
# \brief Indipencene copula.


class IndepCopula(CopulaBase):
    def __init__(self):
        pass

    def _pdf(self, u, v, rotation_theta=0):
        return np.ones(len(u))
