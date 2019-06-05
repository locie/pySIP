from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

from ..utils.matrices import MultivariateNormal
from ..utils.statistics import autocovf


class sMMALA(object):
    """simplified Manifold Metropolis-Adjusted Langevin Algorithm

    The sMMALA is an improved version of the Metropolis-Hastings algorithm,
    and belongs to the Markov Chains Monte Carlo family of samplers. sMMALA
    uses the gradient and the Hessian of the log-posterior.

    Attributes
    ----------
    _lp : function
        function returning the Negative logarithm posterior distribution
    _par : object
        An instance of Parameter (see bopt/parameters.py)
    _M : int
        Number of Markov chains
    _Np : int
        Number of free parameters
    _N : int
        Number of samples by Markov chains
    _Ip : array_like
        Identity matrix of size `_Np`
    _acceptance : array_like
        Store for each iteration the accept/reject flag
    step-size : float or array_like
        Step-size of the proposal distribution
    warmup : int
        Number of iterations to discard as warm-up
    trace : array_like
        Store the trace of the Markov chains

    Methods
    -------
    sampling
        Sample the posterior distribution function
    _run_sampler
        Run the sMMALA sampler
    _dual_averaging
        Estimate the step-size with the primal dual averaging
    acceptance_rate
        Compute the acceptance rate in percent
    iact
        Estimate the Integrated AutoCorrelation Time
    psrf
        Compute the Potential Scale Reduction Factor
    ess
        Estimate the Effective Sample Size
    diagnostic
        Display the diagnostic test results after calling `sampling`
    plot_trace
        Plot the trace of a specific parameter
    _estimate_warmup
        Estimate the number of samples to discard as warm-up
    remove_chains
        Remove undesirable Markov chains from self.trace

    Notes
    -----
    The diagnostic test require to run M >= 2 Markov Chains of N samples,
    for each `_Np` parameter. As a good practice, the Markov chains should
    be initialized with different initial values.

    References
    ----------
    Papamarkou, T., Lindo, A. and Ford, E.B., 2016. Geometric adaptive
    Monte Carlo in random environment. arXiv preprint arXiv:1608.07986.

    Dahlin, J., 2016. Accelerating Monte Carlo methods for Bayesian
    inference in dynamical models (Vol. 1754).
    Linköping UniversityElectronic Press.

    MD Hoffman, A Gelman, 2014. The No-U-Turn sampler: adaptively
    setting path lengths in Hamiltonian Monte Carlo. Journal of Machine
    Learning Research, 2014
    """

    def __init__(self, ln_posterior=None, parameters=None):
        """Initialisation of the sMMALA sampler

        Parameter
        ---------
        ln_posterior : function
            function returning the Negative logarithm posterior distribution
        parameters : object
            An instance of Parameter (see bopt/parameters.py)
        """

        if ln_posterior is None:
            raise TypeError("A log-posterior function must be specified")

        if parameters is None:
            raise TypeError("A Parameter instance is required")

        self._lp = ln_posterior
        self._par = parameters

    def sampling(self, dt, u, u1, y, options=None):
        """Sample the posterior distribution function with the sMMALA

        The optimal acceptance rate of the sMMALA is between 0.574 and the
        step-size for the optimal scaling result at stationnarity is
        :math:`\\epsilon = 1.65^{2} d^{-1/3}`

        Parameters
        ----------
        dt : array_like
            sampling time vector (N-1,)
        cst_dt : bool
            flag for constant sampling time
        y : array_like
            output data
        u : array_like
            zero order hold input data
        u1 : array_like
            first order hold input data
        hold : str
            order hold input assumption
        options : dict
            `n_chains` : int (default: 4)
                number of Markov chains
            `n_samples` : int (default: 3000)
                number of samples by Markov chains
            `step_size` : float, array_like (default: None)
                step-size of the proposal distribution, by default the
                step-size is estimated with a tuning-run
            `acceptance_rate` : float
                acceptance rate in the stationary distribution
            `theta_init` : str (TODO)
                If `theta_init` is set to 'prior', the initial parameters are
                randomly sampled from their respective prior distributions
            `prob_prior` : float
                Use only `prob_prior` percent of the prior distribution to
                sample the initial parameters with `theta_init`
            `rcond` : float
                If the reciprocal condition number of the Hessian matrix is
                below `rcond`, a pseudo inverse is used
            `alpha_SoftAbs` : float
                soft absolute eigenvalues of the hessian at 1/`alpha_SoftAbs`
            `n_adapt` : int
                maximum number of iterations for tuning the step-size
            `lb_step_size` : float
                lower bound of the step-size
            `ub_step_size` : float
                upper bound of the step-size
            `t0` : float
                primal dual averaging parameter, iterations bias to limit
                the behavior of the algorithm in the early iterations
            `gamma` : float
                primal dual averaging parameter, control the shrinkage
                toward the prior mean `mu`
            `kappa` : float
                primal dual averaging parameter, control the step-size
                of the algorithm
            `mu` : float
                primal dual averaging parameter, prior mean of the step-size
        """

        # get options and set default options if not specified
        if options is None:
            options = {}
        else:
            options = dict(options)

        options.setdefault("n_chains", 4)
        options.setdefault("n_samples", 3000)

        n_chains = options["n_chains"]
        if not isinstance(n_chains, int):
            raise TypeError("The number of chains must be an integer")

        if n_chains < 0:
            raise ValueError("the number of chains must be positive")

        n_samples = options["n_samples"]
        if not isinstance(n_samples, int):
            raise TypeError("The number of iterations must be an integer")

        if n_samples < 0:
            raise ValueError("the number of iterations must be positive")

        if n_samples < 250:
            raise ValueError("At least 250 samples are required")

        # set default warmup to half the number of iterations
        self.warmup = int(0.5 * n_samples)

        self._M = n_chains
        self._Np = len(self._par.eta_free)
        self._N = n_samples
        self._Ip = np.eye(self._Np)

        options.setdefault("step_size", None)
        options.setdefault("acceptance_rate", 0.574)
        options.setdefault("theta_init", "prior")
        options.setdefault("prob_prior", 0.95)
        options.setdefault("alpha_SoftAbs", 1e6)
        options.setdefault("n_adapt", 3000)
        options.setdefault("lb_step_size", 0.3)
        options.setdefault("ub_step_size", 3.0)
        options.setdefault("t0", 15.0)
        options.setdefault("gamma", 0.05)
        options.setdefault("kappa", 0.9)
        options.setdefault("mu", 1.65**2 / self._Np**(1 / 3))

        if not isinstance(options["n_adapt"], int):
            raise TypeError("The number of iterations for the step-size "
                            " adaptation must be an integer")

        if options["n_adapt"] < 0:
            raise TypeError("The number of iterations for the step-size "
                            " adaptation must be positive")

        if not 0.0 < options["acceptance_rate"] < 1.0:
            raise TypeError("the acceptance rate must be between 0 and 1")

        if not 0.0 < options["prob_prior"] < 1.0:
            raise TypeError(
                "the probability area of the prior must be between 0 and 1")

        if options["alpha_SoftAbs"] < 0.0:
            raise TypeError("the hardness parameter of the SoftAbs metric "
                            "must be positive")

        if options["t0"] < 0.0:
            raise TypeError("the tuning parameter `t0` of the primal dual "
                            "averaging method must be positive")

        if not 0.0 < options["kappa"] < 1.0:
            raise TypeError("the tuning parameter `kappa` of the primal dual "
                            "averaging method must be between 0 and 1")

        if options["mu"] < 0.0:
            raise TypeError("the tuning parameter `mu` of the primal dual "
                            "averaging method must be positive")

        if options["lb_step_size"] > options["ub_step_size"]:
            raise TypeError("the lower bound of the step-size must be "
                            "inferior to the upper bound")

        step_size = options["step_size"]
        if step_size is None:
            # random initial parameters in unconstrained space
            # eta0 = self._par.eta_free
            # self._par.theta = self._par.prior_init(0.95)

            print(f"Tuning of the step length, acceptance rate target:"
                  f" {options['acceptance_rate']}\n")
            self.step_size = self._dual_averaging(dt, u, u1, y, options)
            print(f"step length: {self.step_size}\n")
            # self._par.eta = eta0
        else:
            if isinstance(step_size, (tuple, list, np.ndarray)):
                step_size_shape = np.shape(step_size)
                if len(step_size_shape) == 1:
                    if step_size_shape[0] != self._Np:
                        raise TypeError(f"The step size vector must have"
                                        f" {self._Np} elements")
                else:
                    if step_size_shape[0] and step_size_shape[1] != self._Np:
                        raise TypeError(f"The step size matrix must have"
                                        f" {self._Np} rows and columns")

            if np.any(step_size) < 0:
                raise TypeError("the step size values must be positive")

            self.step_size = step_size

        # memory allocation for the traces and the acceptance rate
        self.trace = np.empty((n_chains, self._Np + 1, n_samples))
        self._acceptance = np.empty((n_chains, n_samples))

        out = Parallel(n_jobs=-1, backend='loky')(delayed(self._parallel_sampling)(dt, u, u1, y, options, _iter) for _iter in range(n_chains))
        for i in range(n_chains):
            self.trace[i, :, :] = out[i][0]
            self._acceptance[i, :] = out[i][1]
        # for n in out:
        #     self.trace[n, :, :] = n[0]
        #     self._acceptance[n, :] = n[1]
        # run the sampler n_chains times
        # for i in range(n_chains):
        #     # random initial parameters in unconstrained space
        #     # self._par.eta = eta0
        #     # options["index"] = i
        #     self._par.theta = self._par.prior_init(options["prob_prior"])
        #     # self._par.eta = -2. + 4. * np.random.random(self._Np)
        #     out = self._run_sampler(dt, u, u1, y, options)
        #     self.trace[i, :, :] = out[0]
        #     self._acceptance[i, :] = out[1]

    def _parallel_sampling(self, dt, u, u1, y, options, _iter):
        self._par.theta = self._par.prior_init(options["prob_prior"])
        out = self._run_sampler(dt, u, u1, y, options, _iter)
        return out

    def _run_sampler(self, dt, u, u1, y, options, _iter):
        """Run the sMMALA sampler

        Parameters
        ----------
        dt : array_like
            sampling time vector (N-1,)
        cst_dt : bool
            flag for constant sampling time
        y : array_like
            output data
        u : array_like
            zero order hold input data
        u1 : array_like
            first order hold input data
        hold : str
            order hold input assumption
        options : dict
            options of the mcmc sampler

        Return
        ------
        trace : array_like
            Markov Chain traces of the parameters and the posterior
        acceptance : array_like
            Store the accept/reject flag

        """
        n_samples = options.get("n_samples")

        inv_epsilon = self._Ip / self.step_size

        # Allocate memory for the output
        trace = np.empty((self._Np + 1, n_samples))
        acceptance = np.zeros(n_samples)

        # compute posterior information at initial parameter values
        posterior, gradient, hessian = self._lp(self._par.eta_free,
                                                dt, u, u1, y)

        # create instances of Multivariate Normal distribution
        mvn = MultivariateNormal(self._Np, options.get("alpha_SoftAbs"))
        mvnP = MultivariateNormal(self._Np, options.get("alpha_SoftAbs"))

        mvn.precision = hessian @ inv_epsilon
        mvn.mean = self._par.eta_free - 0.5 * mvn.covariance @ gradient

        # save initial state of the Markov chain
        trace[:self._Np, 0] = self._par.theta_free
        trace[-1, 0] = posterior
        acceptance[0] = 1

        pbar = tqdm(range(1, n_samples), position=_iter, desc='chains n°' + str(_iter))
        for i in pbar:
            # for i in range(1, n_samples):
            # Generate samples from the proposal distribution
            etaP = mvn.random()

            # Evaluate posterior distribution at the proposed parameter
            posteriorP, gradient, hessian = self._lp(etaP, dt, u, u1, y)

            # Distribution of the proposed parameter
            mvnP.precision = hessian @ inv_epsilon
            mvnP.mean = etaP - 0.5 * mvnP.covariance @ gradient

            # forward state transition probability: p(proposed|current)
            eta_etaP = mvn.log_pdf(etaP)

            # backward state transition probability: p(current|proposed)
            etaP_eta = mvnP.log_pdf(self._par.eta_free)

            # acceptance probability
            alpha = np.exp(posterior - posteriorP + etaP_eta - eta_etaP)

            # Accept / Reject
            if np.random.random() < np.min((1.0, alpha)):
                acceptance[i] = 1
                self._par.eta = etaP
                posterior = deepcopy(posteriorP)
                mvn = deepcopy(mvnP)

            # save current state of the Markov chain
            trace[:self._Np, i] = self._par.theta_free
            trace[-1, i] = posterior

            # print progress of the Markov Chain
            # pbar.set_description(f"Markov chain {index + 1}/{self._M}")

        return trace, acceptance

    def _dual_averaging(self, dt, u, u1, y, options):
        """Tune the step size to match the desired acceptance rate

        Parameters
        ----------
        dt : array_like
            sampling time vector (N-1,)
        cst_dt : bool
            flag for constant sampling time
        y : array_like
            output data
        u : array_like
            zero order hold input data
        u1 : array_like
            first order hold input data
        hold : str
            order hold input assumption
        options : dict
            options of the mcmc sampler

        Return
        ------
        step_size_bar : float
            estimated step size of the proposal distribution

        References
        ----------
        MD Hoffman, A Gelman, 2014. The No-U-Turn sampler: adaptively
        setting path lengths in Hamiltonian Monte Carlo. Journal of Machine
        Learning Research, 2014

        """
        acceptance_rate = options.get("acceptance_rate")
        n_adapt = options.get("n_adapt")
        lb_step_size = options.get("lb_step_size")
        ub_step_size = options.get("ub_step_size")
        t0 = options.get("t0")
        gamma = options.get("gamma")
        kappa = options.get("kappa")
        mu = options.get("mu")

        # step size
        step_size = 1.0

        # compute posterior information at initial parameter values
        posterior, gradient, hessian = self._lp(self._par.eta_free,
                                                dt, u, u1, y)

        # create instances of Multivariate Normal distribution
        mvn = MultivariateNormal(self._Np, options.get("alpha_SoftAbs"))
        mvnP = MultivariateNormal(self._Np, options.get("alpha_SoftAbs"))

        mvn.precision = hessian @ (self._Ip / step_size)
        mvn.mean = self._par.eta_free - 0.5 * mvn.covariance @ gradient

        # Initialize the primal-dual averaging
        H_bar = 0.0
        step_size_bar_u = np.log((step_size - lb_step_size)
                                 / (ub_step_size - step_size))

        conv = 0
        for i in range(1, n_adapt):
            # Generate samples from the proposal distribution
            etaP = mvn.random()

            # Evaluate posterior distribution at the proposed parameter
            posteriorP, gradient, hessian = self._lp(etaP, dt, u, u1, y)

            # Distribution of the proposed parameter
            mvnP.precision = hessian @ (self._Ip / step_size)
            mvnP.mean = etaP - 0.5 * mvnP.covariance @ gradient

            # forward state transition probability: p(proposed|current)
            eta_etaP = mvn.log_pdf(etaP)

            # backward state transition probability: p(current|proposed)
            etaP_eta = mvnP.log_pdf(self._par.eta_free)

            # acceptance probability
            alpha = np.exp(posterior - posteriorP + etaP_eta - eta_etaP)
            beta = np.min((1.0, alpha))

            # Accept / Reject
            if np.random.random() < beta:
                self._par.eta = etaP
                posterior = deepcopy(posteriorP)
                mvn = deepcopy(mvnP)

            # Adapt the step length with primal-dual averaging
            # average error
            delta = 1. / (i + t0)
            H_bar = (1. - delta) * H_bar + delta * (acceptance_rate - beta)

            # unconstrained step size
            step_size_u = mu - np.sqrt(i) / gamma * H_bar

            # constrained step size
            step_size = lb_step_size + ((ub_step_size - lb_step_size)
                                        / (1. + np.exp(-step_size_u)))

            # update step size mean in unconstrained space
            delta = i ** -kappa
            step_size_bar_u = (delta * step_size_u
                               + (1. - delta) * step_size_bar_u)

            # constrained step size mean
            step_size_bar = lb_step_size + ((ub_step_size - lb_step_size)
                                            / (1.0 + np.exp(-step_size_bar_u)))

            if np.abs(H_bar) < 0.01:
                conv += 1
            else:
                conv = 0

            if conv == 100:
                print("Convergence criterion satisfied")
                break

            # print progress of the Markov Chain
            print(f"ln-posterior: {posterior:.3f}"
                  f" | step-length: {step_size_bar:.3f} | Hbar: {H_bar:.3f}")

        return step_size_bar

    @property
    def acceptance_rate(self):
        """Compute the acceptance rate of the Markov chains

        Return
        ------
        float
            acceptance rate in percent for each Markov chains
        """

        return (self._acceptance[:, self.warmup:].sum(axis=1)
                / (self._N - self.warmup)) * 100

    def iact(self):
        """Integrated Auto-Correlation Time (IACT)

        The IACT estimates the number of iterations between two
        uncorrelated samples. The IACT is computed with mutliple
        Markov chains in order to reduce the variance of the estimation.

        Return
        ------
        iact: array_like
            Estimated IACT for each parameters

        Notes
        -----
        It's recommended to use at least 4 Markov chains

        References
        ----------
        section 11.5 of Gelman, A., Stern, H.S., Carlin, J.B., Dunson, D.B.,
        Vehtari, A. and Rubin, D.B., 2013.
        Bayesian data analysis. Chapman and Hall/CRC.

        Stan Reference Manual 2.18.1
        section Posterior Analysis / Effective Sample Size
        """

        if self._M < 2:
            raise ValueError("At least two Markov chains are required")

        # remove warm-up samples
        chains = self.trace[:, :, self.warmup:]

        # split each chains into a first and second half of the same dimension
        N = int((self._N - self.warmup) / 2)
        M = self._M * 2
        chains = np.concatenate((chains[:, :, -2 * N: -N], chains[:, :, -N:]))

        # compute the estimate IACT for each parameter
        iact_hat = np.empty(self._Np + 1)
        for i in range(self._Np + 1):
            # compute the autocovariance function for the M chains
            acov = np.asarray([autocovf(chains[m, i, :]) for m in range(M)])

            # between chain variance divided by N
            B_over_N = np.var(chains[:, i, :].mean(axis=1), ddof=1)

            # within chain variance (note: the autocorrelation function at the
            # lag 0 is equal to the sample variance of the chain)
            W = np.mean(acov[:, 0] * N / (N - 1.))

            # mixture of the within-chain and cross-chain sample variances
            var_hat = (1 - 1 / N) * W + B_over_N

            # combined autocorrelation
            rho_hat = 1. - (W - np.mean(acov * N / (N - 1), axis=0)) / var_hat

            # Geyer's initial positive sequence
            tp = 0
            rho_hat_trunc = []
            while tp < (N - 2):
                paired_sum = rho_hat[tp] + rho_hat[tp + 1]
                if paired_sum >= 0:
                    rho_hat_trunc.append(paired_sum)
                else:
                    break
                tp += 2

            # Geyer's initial monotone sequence
            for tm in range(1, len(rho_hat_trunc)):
                if rho_hat_trunc[tm] > rho_hat_trunc[tm - 1]:
                    rho_hat_trunc[tm] = rho_hat_trunc[tm - 1]

            iact_hat[i] = -1. + 2. * np.sum(rho_hat_trunc)

        return iact_hat

    def ess(self):
        """Compute the effective sample size (ESS)

        The ESS represents the number of uncorrelated samples in the Markov
        chains. The variance of the Monte Carlo estimator is proportional to
        1 / sqrt(ESS) instead of 1 / sqrt(N) since the samples from the Markov
        chains are not independent and identically distributed.

        Return
        ------
        ess: array_like
            Estimate ESS for each parameters

        Notes
        -----
        For better accuracy use at least 4 Markov chains

        References
        ----------
        section 11.5 of Gelman, A., Stern, H.S., Carlin, J.B., Dunson, D.B.,
        Vehtari, A. and Rubin, D.B., 2013.
        Bayesian data analysis. Chapman and Hall/CRC.

        Stan Reference Manual 2.18.1
        section Posterior Analysis / Effective Sample Size
        """

        # get integrated autocorrelation time
        iact_hat = self.iact()

        return (self._M * (self._N - self.warmup)) / iact_hat

    def psrf(self):
        """Potential Scale Reduction Factor (Gelman-Rubin diagnostic)

        We consider that the Markov chains has reach its stationary
        distribution if R_hat < 1.01

        Return
        ------
        R_hat: array_like
            Potential scale reduction factor for each parameters

        Notes
        -----
        For better accuracy use at least 4 Markov chains

        References
        ----------
        Section 11.4 of Gelman, A., Stern, H.S., Carlin, J.B., Dunson, D.B.,
        Vehtari, A. and Rubin, D.B., 2013.
        Bayesian data analysis. Chapman and Hall/CRC.
        """

        if self._M < 2:
            raise ValueError("At least two Markov chains are required")

        # remove warm-up samples
        chains = self.trace[:, :, self.warmup:]

        # split each chains into a first and second half of the same dimension
        N = int((self._N - self.warmup) / 2)
        chains = np.concatenate((chains[:, :, -2 * N: -N], chains[:, :, -N:]))

        # compute the PSRF for each parameter
        R_hat = np.empty(self._Np + 1)
        for i in range(self._Np + 1):
            # between chain variance divided by N
            B_over_N = np.var(chains[:, i, :].mean(axis=1), ddof=1)

            # within-chain variance
            W = np.mean(np.var(chains[:, i, :], axis=1, ddof=1))

            # mixture of the within-chain and cross-chain sample variances
            var_hat = (1 - 1 / N) * W + B_over_N

            R_hat[i] = np.sqrt(var_hat / W)

        return R_hat

    def diagnostic(self, warmup_estimation=True):
        """Display summary / diagnostic of the sampling

        Estimate the warm-up if required and display:
            - names of the free parameters
            - estimated mean
            - standard error of the estimated mean
            - potential scale reduction factor
            - integrated autocorrelation time
            - effective sample size
        """

        if warmup_estimation:
            self._estimate_warmup()

        print("summary")
        print("-------")
        print(f"{self._M} chains of {self._N} samples have been simulated and"
              f" {self.warmup} samples have been discarded for the warmup.\n")

        names = self._par.names_free
        names.append("ln_post")
        psrf = self.psrf()
        iact = self.iact()
        ess = self._M * (self._N - self.warmup) / iact
        mean = self.trace[:, :, self.warmup:].mean(axis=(0, 2))
        se = self.trace[:, :, self.warmup:].std(axis=(0, 2)) / np.sqrt(ess)

        print("{0:^12} {1:^12} {2:^12} {3:^12} {4:^12} {5:^12}".format(
            "names", "mean(\u03B8)", "se(\u03B8)", "psrf", "iact", "ess"))
        print("{0:^12} {0:^12} {0:^12} {0:^12} {0:^12} {0:^12}".format(
            "-----------"))
        sfor = "{0:^12} {1:^12.3e} {2:^12.3e} {3:^12.3f} {4:^12.3f} {5:^12.3e}"
        for i in range(self._Np + 1):
            print(sfor.format(names[i], mean[i], se[i],
                              psrf[i], iact[i], ess[i]))

    def plot_trace(self, param_name=None, remove_warmup=False):
        """plot the Markov chains traces

        Parameters
        ----------
        param_name : str
            a parameter name belonging to self._par.names_free
        remove_warmup : bool
            if set to True, the warm-up is discarded

        Return
        ------
        plot
        """

        if param_name is None:
            raise TypeError("A parameter name is required")

        if not isinstance(param_name, str):
            raise TypeError("The parameter name must be a string")

        if param_name == "ln_post":
            idx = -1
        else:
            if param_name not in self._par.names_free:
                raise TypeError(f"{param_name} is not in the parameter list "
                                f"{self._par.names_free}")
            idx = self._par.names_free.index(param_name)

        if not remove_warmup:
            warmup = 0
        else:
            warmup = self.warmup

        for m in range(self._M):
            plt.plot(self.trace[m, idx, warmup:])

        plt.show()

    def _estimate_warmup(self):
        """Estimate the number of samples to discard as warm-up

        Find the number of samples to discard in order to maximize the
        effective sample size.

        Return
        ------
        self.warmup : int

        """
        # find the highest sum of effective sample size
        self.warmup = 0
        idx = 0
        max_ess = np.sum(self.ess())
        i = 1
        while i < self._N - 100:
            self.warmup = i
            sum_ess = np.sum(self.ess())
            if sum_ess > max_ess:
                max_ess = sum_ess
                idx = i
            i += 1

        # set estiamted warmup
        self.warmup = idx

        R_hat = self.psrf()
        if np.any(R_hat > 1.01):
            print("\nThe Markov chains corresponding to the maxium effective "
                  "sample has not reach the stationary distribution. A "
                  "visual inspection of the traces is required.\n")

    def remove_chains(self, index):
        """Remove undesirable Markov Chains

        It is possible that a Markov chain doesn't reach the stationay
        distribution and has to be discared for estimating the posterior
        distribution.

        Parameter
        ---------
        index : int or array_like
            If multiple chains have to be discarded, `index` is an array_like
            with multiple indices

        Notes
        -----
        Once a chain has been discared, it cannot be recovered
        """

        index = np.atleast_1d(index)
        if len(index.shape) != 1:
            raise TypeError("index must be a scalar or a vector")
        elif index.shape[0] > self._M:
            raise TypeError("Cannot remove more chains than available")

        index_bool = np.full(self._M, True, dtype=bool)
        index_bool[index] = False

        self.trace = self.trace[index_bool, :, :]
        self._M = self.trace.shape[0]
