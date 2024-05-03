import itertools
from typing import Optional

import numpy as np
import numpy.typing as npt

from sgrmemb.densitygen import DistributionFunction
from sgrmemb.em import expectation


class ModelContainer:
    def __init__(self, eta_0: float = 0.5) -> None:

        self.categories: set[str] = {"memb", "field"}
        self.spaces: set[str] = set()
        self.models: dict[tuple[str, str], DistributionFunction] = dict()
        self.data: dict[tuple[str, str], npt.ArrayLike] = dict()
        self._eta: list[float] = [eta_0]
        self.prob_memb: npt.ArrayLike = None
        self.prob_field: npt.ArrayLike = None

    @property
    def eta(self) -> float:
        return self._eta[-1]

    @property
    def diff_eta(self) -> float:
        if len(self._eta) < 2:
            return 9.9e99
        else:
            return np.abs(self._eta[-1] - self._eta[-2])

    def modify_parameter(
        self,
        category: str,
        space: str,
        model: Optional[DistributionFunction] = None,
        data: Optional[npt.ArrayLike] = None,
    ) -> None:
        """
        Add a model or data to the model container.
        """

        # Check if category is valid
        if category not in self.categories:
            raise ValueError(
                f"Category {category} not valid. Valid categories are {self.categories}"
            )

        # Add space to list of spaces
        self.spaces.add(space)

        # Add model
        if model is not None:
            self.models[(space, category)] = model

        # Add data
        if data is not None:
            self.data[(space, category)] = data

    def modify_parameters(
        self,
        categories: list[str],
        spaces: list[str],
        models: Optional[list[DistributionFunction]] = None,
        data: Optional[list[npt.ArrayLike]] = None,
    ) -> None:
        """
        Modify multiple models or data.
        """
        # check if categories and spaces have the same length
        if len(categories) != len(spaces):
            raise ValueError(
                f"categories and spaces must have the same length. "
                f"Found {len(categories)} categories and {len(spaces)} spaces"
            )


        for cat, space, model, data in zip(categories, spaces, models, data):
            self.modify_parameter(category=cat, space=space, model=model, data=data)
        

    def run_expectation(
        self, n_iterations: int = 1, exclude_space: Optional[set[str]] = None
    ) -> None:
        """
        Run the expectation step of the EM algorithm.

        Parameters
        ----------
        n_iterations : int
            Number of iterations to run the expectation step.
        exclude_space : set[str]
            Spaces to exclude from the expectation step.

        """

        # Check if exclude_space has valid values
        if exclude_space is not None:
            if not exclude_space.issubset(self.spaces):
                raise ValueError(
                    f"Spaces {exclude_space} not valid. Valid spaces are {self.spaces}"
                )

        active_spaces = (
            self.spaces if exclude_space is None else self.spaces - exclude_space
        )
        active_keys = [key for key in itertools.product(active_spaces, self.categories)]

        # Check data and models
        if not all(key in active_keys for key in self.models.keys()) and all(
            key in active_keys for key in self.data.keys()
        ):
            raise ValueError(
                f"Models and data must be provided for all spaces in {active_spaces}"
            )

        q_memb_i, q_field_i, eta = expectation(
            memb_models=[self.models[k] for k in active_keys if k[1] == "memb"],  #
            field_models=[self.models[k] for k in active_keys if k[1] == "field"],
            data_memb=[self.data[k] for k in active_keys if k[1] == "memb"],
            data_field=[self.data[k] for k in active_keys if k[1] == "field"],
            eta_0=self._eta[-1],
            n_iterations=n_iterations,
        )

        self._eta.append(eta)
        self.prob_memb = q_memb_i
        self.prob_field = q_field_i

    def get_data_as_dict(self) -> dict[str, npt.ArrayLike]:
        """
        Return a dictionary with the keys in the form `space_category_i` and the values the data in numpy format.
        Notice that `i` is the index of the array in the last dimension.
        The output dictionary is suitable for saving the data in a pandas dataframe or in an Astropy Table.
        """

        output_dict = dict()
        keys = [key for key in itertools.product(self.spaces, self.categories)]
        aux_dict = {
            "_".join(key): self.data[key] for key in keys
        }  # Save the data in a dictionary

        # split the data in the last dimension of the array. Example: (N,2) -> (N,)
        for key in aux_dict.keys():
            splitted_array = np.split(aux_dict[key], aux_dict[key].shape[-1], axis=-1)
            for i, arr in enumerate(splitted_array):
                output_dict[f"{key}_{i}"] = np.squeeze(arr)

        # Add probability of membership and field
        if self.prob_memb is not None and self.prob_field is not None:
            output_dict["prob_memb"] = self.prob_memb
            output_dict["prob_field"] = self.prob_field

        return output_dict
