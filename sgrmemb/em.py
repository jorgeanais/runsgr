import logging
from typing import Callable, Iterable

import numpy as np
import numpy.typing as npt


DistributionFunction = Callable[[npt.NDArray], npt.NDArray]
GeneratorFunction = Callable[
    [npt.NDArray, npt.NDArray], DistributionFunction
]  # TODO: check if this is the correct type


def expectation(
    memb_models: Iterable[DistributionFunction],
    field_models: Iterable[DistributionFunction],
    data_memb: Iterable[npt.NDArray[np.float64]],
    data_field: Iterable[npt.NDArray[np.float64]],
    eta_0: float = 0.5,
    n_iterations: int = 1,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
    """
    Compute membership probabilities.
    If n_iterations > 1, the membership probabilities are computed iteratively.
    It returns the last computed value of the probability of a star of been consider member and field,
    and the fraction of members compared to the total sample (eta).
    """

    logging.info("Running expectation step")

    eta = eta_0
    eta_ = []

    prob_xi_memb = np.prod([mm(X) for mm, X in zip(memb_models, data_memb)], axis=0)
    prob_xi_field = np.prod([fm(X) for fm, X in zip(field_models, data_field)], axis=0)

    for i in range(n_iterations):
        total_likelihood = eta * prob_xi_memb + (1 - eta) * prob_xi_field
        q_memb_i = eta * prob_xi_memb / total_likelihood
        q_field_i = (1 - eta) * prob_xi_field / total_likelihood
        eta = np.average(q_memb_i)
        eta_.append(eta)

    return q_memb_i, q_field_i, eta_[-1]
