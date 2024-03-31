
import numpy as np
import pandas as pd
import warnings
import logging
from optuna.trial import Trial
from optuna.study import Study, StudyDirection
from optuna.integration import OptunaSearchCV
from optuna import samplers, storages, pruners, distributions
from optuna.storages._heartbeat import is_heartbeat_enabled
from optuna.distributions import BaseDistribution, IntDistribution, FloatDistribution, CategoricalDistribution
from typing import Sequence, Union, Dict, Any, Optional, Tuple
from numbers import Number
from . import k_space
from Util import Integer, Real, Categorical


_logger = logging.getLogger(__name__)

class KSpaceTrial(Trial):
    def __init__(self, study: "optuna.study.Study", trial_id: int, kspace: KSpace) -> None:
        super().__init__(study, trial_id)
        self.kspace = kspace
    
    def _suggest(self, name: str, distribution: BaseDistribution) -> Any:
        storage = self.storage
        trial_id = self._trial_id
        trial = self._get_latest_trial()

        if name in trial.distributions:
            # No need to sample if already suggested.
            distributions.check_distribution_compatibility(trial.distributions[name], distribution)
            param_value = trial.params[name]
        else:
            if self._is_fixed_param(name, distribution):
                param_value = self._fixed_params[name]
                k_param_value = param_value
                self.set_user_attr(name + "_kx", None)
            elif distribution.single():
                param_value = distributions._get_single_value(distribution)
                k_param_value = self.kspace.kmap(name, param_value)
                self.set_user_attr(name + "_kx", param_value)
            elif self._is_relative_param(name, distribution):
                param_value = self.relative_params[name]
                k_param_value = param_value
                self.set_user_attr(name + "_kx", None)
            else:
                study = pruners._filter_study(self.study, trial)
                param_value = self.study.sampler.sample_independent(
                    study, trial, name, distribution
                )
                k_param_value = self.kspace.kmap(name, param_value)
                self.set_user_attr(name + "_kx", param_value)

            # `param_value` is validated here (invalid value like `np.nan` raises ValueError).
            # print(f"!!!!!!!!!!!!11: param={name}, x={param_value}, kmapped={k_param_value}")
            param_value_in_internal_repr = distribution.to_internal_repr(k_param_value if k_param_value is not None else param_value)
            storage.set_trial_param(trial_id, name, param_value_in_internal_repr, distribution)

            self._cached_frozen_trial.distributions[name] = distribution
            self._cached_frozen_trial.params[name] = k_param_value
        return k_param_value

class KSpaceStudy(Study):
    def __init__(
        self,
        study_name: str,
        storage: Union[str, storages.BaseStorage],
        search_space: Dict[str, distributions.BaseDistribution],
        sampler: Optional[samplers.BaseSampler] = None,
        pruner: Optional[pruners.BasePruner] = None,
        k: Union[Number, dict] = None,
        k_space_ver: int = 1
    ) -> None:
        super().__init__(study_name, storage, sampler, pruner)
        self.k = k
        cls: k_space.KSpace = None

        if k_space_ver == 1:
            cls = k_space.KSpace
        else:
            cls = getattr(k_space, "KSpaceV" + str(k_space_ver), None)
            if cls is None:
                raise RuntimeError(f"Invalid kspace implementation version: {k_space_ver}")

        self.kspace = cls(self._encode_k_search_space(search_space), k, x_in_search_space=True)
    
    def _encode_k_search_space(self, search_space: Dict[str, distributions.BaseDistribution]) -> dict:
        space = {}
        for k, v in search_space.items():
            if isinstance(v, IntDistribution):
                space[k] = Integer(v.low, v.high, name=k)
            elif isinstance(v, FloatDistribution):
                space[k] = Real(v.low, v.high, name=k)
            elif isinstance(v, CategoricalDistribution):
                space[k] = Categorical(list(v.choices), name=k)
            else:
                raise ValueError(f"search space contains unsupported type for '{k}': {type(v)}")
        return space

    def ask(
        self, fixed_distributions: Optional[Dict[str, BaseDistribution]] = None
    ) -> KSpaceTrial:
        """Create a new trial from which hyperparameters can be suggested.

        This method is part of an alternative to :func:`~optuna.study.Study.optimize` that allows
        controlling the lifetime of a trial outside the scope of ``func``. Each call to this
        method should be followed by a call to :func:`~optuna.study.Study.tell` to finish the
        created trial.

        .. seealso::

            The :ref:`ask_and_tell` tutorial provides use-cases with examples.

        Example:

            Getting the trial object with the :func:`~optuna.study.Study.ask` method.

            .. testcode::

                import optuna


                study = optuna.create_study()

                trial = study.ask()

                x = trial.suggest_float("x", -1, 1)

                study.tell(trial, x**2)

        Example:

            Passing previously defined distributions to the :func:`~optuna.study.Study.ask`
            method.

            .. testcode::

                import optuna


                study = optuna.create_study()

                distributions = {
                    "optimizer": optuna.distributions.CategoricalDistribution(["adam", "sgd"]),
                    "lr": optuna.distributions.FloatDistribution(0.0001, 0.1, log=True),
                }

                # You can pass the distributions previously defined.
                trial = study.ask(fixed_distributions=distributions)

                # `optimizer` and `lr` are already suggested and accessible with `trial.params`.
                assert "optimizer" in trial.params
                assert "lr" in trial.params

        Args:
            fixed_distributions:
                A dictionary containing the parameter names and parameter's distributions. Each
                parameter in this dictionary is automatically suggested for the returned trial,
                even when the suggest method is not explicitly invoked by the user. If this
                argument is set to :obj:`None`, no parameter is automatically suggested.

        Returns:
            A :class:`~optuna.trial.Trial`.
        """

        if not self._thread_local.in_optimize_loop and is_heartbeat_enabled(self._storage):
            warnings.warn("Heartbeat of storage is supposed to be used with Study.optimize.")

        fixed_distributions = fixed_distributions or {}
      
        # Sync storage once every trial.
        self._thread_local.cached_all_trials = None

        trial_id = self._pop_waiting_trial_id()
        if trial_id is None:
            trial_id = self._storage.create_new_trial(self._study_id)
        trial = KSpaceTrial(self, trial_id, self.kspace)

        for name, param in fixed_distributions.items():
            trial._suggest(name, param)

        return trial
    
    @classmethod
    def create_study(
        cls,
        search_space: dict,
        k:  Union[Number, dict] = None,
        storage: Union[str, storages.BaseStorage, None] = None,
        sampler: Optional[samplers.BaseSampler] = None,
        pruner: Optional[pruners.BasePruner] = None,
        study_name: Optional[str] = None,
        direction: Union[str, StudyDirection, None] = None,
        load_if_exists: bool = False,
        directions: Optional[Sequence[Tuple[str, StudyDirection]]] = None,
        k_space_ver: int = 1
    ) -> 'KSpaceStudy':
        """Create a new :class:`~optuna.study.Study`.

        Example:

            .. testcode::

                import optuna


                def objective(trial):
                    x = trial.suggest_float("x", 0, 10)
                    return x**2


                study = optuna.create_study()
                study.optimize(objective, n_trials=3)

        Args:
            storage:
                Database URL. If this argument is set to None, in-memory storage is used, and the
                :class:`~optuna.study.Study` will not be persistent.

                .. note::
                    When a database URL is passed, Optuna internally uses `SQLAlchemy`_ to handle
                    the database. Please refer to `SQLAlchemy's document`_ for further details.
                    If you want to specify non-default options to `SQLAlchemy Engine`_, you can
                    instantiate :class:`~optuna.storages.RDBStorage` with your desired options and
                    pass it to the ``storage`` argument instead of a URL.

                .. _SQLAlchemy: https://www.sqlalchemy.org/
                .. _SQLAlchemy's document:
                    https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls
                .. _SQLAlchemy Engine: https://docs.sqlalchemy.org/en/latest/core/engines.html

            sampler:
                A sampler object that implements background algorithm for value suggestion.
                If :obj:`None` is specified, :class:`~optuna.samplers.TPESampler` is used during
                single-objective optimization and :class:`~optuna.samplers.NSGAIISampler` during
                multi-objective optimization. See also :class:`~optuna.samplers`.
            pruner:
                A pruner object that decides early stopping of unpromising trials. If :obj:`None`
                is specified, :class:`~optuna.pruners.MedianPruner` is used as the default. See
                also :class:`~optuna.pruners`.
            study_name:
                Study's name. If this argument is set to None, a unique name is generated
                automatically.
            direction:
                Direction of optimization. Set ``minimize`` for minimization and ``maximize`` for
                maximization. You can also pass the corresponding :class:`~optuna.study.StudyDirection`
                object. ``direction`` and ``directions`` must not be specified at the same time.

                .. note::
                    If none of `direction` and `directions` are specified, the direction of the study
                    is set to "minimize".
            load_if_exists:
                Flag to control the behavior to handle a conflict of study names.
                In the case where a study named ``study_name`` already exists in the ``storage``,
                a :class:`~optuna.exceptions.DuplicatedStudyError` is raised if ``load_if_exists`` is
                set to :obj:`False`.
                Otherwise, the creation of the study is skipped, and the existing one is returned.
            directions:
                A sequence of directions during multi-objective optimization.
                ``direction`` and ``directions`` must not be specified at the same time.

        Returns:
            A :class:`~optuna.study.Study` object.

        See also:
            :func:`optuna.create_study` is an alias of :func:`optuna.study.create_study`.

        See also:
            The :ref:`rdb` tutorial provides concrete examples to save and resume optimization using
            RDB.

        """

        if direction is None and directions is None:
            directions = ["maximize"]
        elif direction is not None and directions is not None:
            raise ValueError("Specify only one of `direction` and `directions`.")
        elif direction is not None:
            directions = [direction]
        elif directions is not None:
            directions = list(directions)
        else:
            assert False

        if len(directions) < 1:
            raise ValueError("The number of objectives must be greater than 0.")
        elif any(
            d not in ["minimize", "maximize", StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE]
            for d in directions
        ):
            raise ValueError(
                "Please set either 'minimize' or 'maximize' to direction. You can also set the "
                "corresponding `StudyDirection` member."
            )

        direction_objects = [
            d if isinstance(d, StudyDirection) else StudyDirection[d.upper()] for d in directions
        ]

        storage = storages.get_storage(storage)
        try:
            study_id = storage.create_new_study(direction_objects, study_name)
        except exceptions.DuplicatedStudyError:
            if load_if_exists:
                assert study_name is not None

                _logger.info(
                    "Using an existing study with name '{}' instead of "
                    "creating a new one.".format(study_name)
                )
                study_id = storage.get_study_id_from_name(study_name)
            else:
                raise

        if sampler is None and len(direction_objects) > 1:
            sampler = samplers.NSGAIISampler()

        study_name = storage.get_study_name_from_id(study_id)
        study = KSpaceStudy(study_name=study_name, storage=storage, sampler=sampler, pruner=pruner, search_space=search_space, k=k, k_space_ver=k_space_ver)

        return study

