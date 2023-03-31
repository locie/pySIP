"""Bayesian Filter template"""


class BayesianFilter:
    """Template for Bayesian filters"""

    def predict(self, **kwargs):
        """State predictive distribution from time t to t+1"""
        raise NotImplementedError

    def update(self, **kwargs):
        """Filtered state distribution at time t"""
        raise NotImplementedError

    def filtering(self, **kwargs):
        """Compute the filtered state distribution and the residuals"""
        raise NotImplementedError

    def smoothing(self, **kwargs):
        """Compute the smoothed state distribution"""
        raise NotImplementedError

    def log_likelihood(self, **kwargs):
        """Evaluate the negative log-likelihood"""
        raise NotImplementedError
