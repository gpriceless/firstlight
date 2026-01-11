"""
Adaptive Parameter Tuning Module.

Provides automatic parameter adjustment when standard parameters
produce poor results:
- Grid search for optimal parameters
- Bayesian optimization for efficient search
- Per-region parameter optimization
- Adaptive threshold adjustment
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging
import uuid

import numpy as np

logger = logging.getLogger(__name__)


class TuningMethod(Enum):
    """Methods for parameter tuning."""
    GRID_SEARCH = "grid_search"      # Exhaustive grid search
    RANDOM_SEARCH = "random_search"  # Random sampling
    BAYESIAN = "bayesian"            # Bayesian optimization
    ADAPTIVE = "adaptive"            # Adaptive threshold adjustment
    EVOLUTIONARY = "evolutionary"    # Evolutionary optimization


@dataclass
class ParameterSpace:
    """
    Defines the search space for a parameter.

    Attributes:
        name: Parameter name
        min_value: Minimum value
        max_value: Maximum value
        step: Step size for grid search (None for continuous)
        default: Default value
        log_scale: Whether to search in log scale
    """
    name: str
    min_value: float
    max_value: float
    step: Optional[float] = None
    default: Optional[float] = None
    log_scale: bool = False

    def get_grid_values(self, n_points: int = 10) -> np.ndarray:
        """Get grid values for this parameter."""
        if self.step is not None:
            values = np.arange(self.min_value, self.max_value + self.step, self.step)
        elif self.log_scale:
            values = np.logspace(
                np.log10(self.min_value),
                np.log10(self.max_value),
                n_points,
            )
        else:
            values = np.linspace(self.min_value, self.max_value, n_points)
        return values

    def sample_random(self) -> float:
        """Sample a random value from this parameter space."""
        if self.log_scale:
            log_min = np.log10(self.min_value)
            log_max = np.log10(self.max_value)
            return float(10 ** np.random.uniform(log_min, log_max))
        return float(np.random.uniform(self.min_value, self.max_value))


@dataclass
class TuningHistory:
    """
    Record of a parameter tuning attempt.

    Attributes:
        trial_id: Unique identifier
        parameters: Parameter values tried
        score: Resulting score
        timestamp: When this trial ran
        duration_ms: Time taken
        metadata: Additional trial information
    """
    trial_id: str
    parameters: Dict[str, float]
    score: float
    timestamp: datetime
    duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trial_id": self.trial_id,
            "parameters": self.parameters,
            "score": round(self.score, 4),
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": round(self.duration_ms, 2),
            "metadata": self.metadata,
        }


@dataclass
class ParameterTuningResult:
    """
    Result of parameter tuning.

    Attributes:
        best_parameters: Optimal parameters found
        best_score: Best score achieved
        history: All trials run
        method: Tuning method used
        total_trials: Number of trials
        total_duration_ms: Total tuning time
        improved: Whether tuning improved over default
        improvement_percent: Percentage improvement
    """
    best_parameters: Dict[str, float]
    best_score: float
    history: List[TuningHistory]
    method: TuningMethod
    total_trials: int
    total_duration_ms: float
    improved: bool
    improvement_percent: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "best_parameters": self.best_parameters,
            "best_score": round(self.best_score, 4),
            "method": self.method.value,
            "total_trials": self.total_trials,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "improved": self.improved,
            "improvement_percent": round(self.improvement_percent, 2),
            "history": [h.to_dict() for h in self.history[-10:]],  # Last 10
            "metadata": self.metadata,
        }


@dataclass
class ParameterTuningConfig:
    """
    Configuration for parameter tuning.

    Attributes:
        method: Tuning method to use
        max_trials: Maximum number of trials
        timeout_ms: Maximum tuning time
        early_stopping_patience: Trials without improvement before stopping
        min_improvement_threshold: Minimum improvement to consider success
        n_initial_random: Random trials before optimization (for Bayesian)
        grid_points_per_param: Grid points per parameter (for grid search)
    """
    method: TuningMethod = TuningMethod.GRID_SEARCH
    max_trials: int = 50
    timeout_ms: float = 60000.0
    early_stopping_patience: int = 10
    min_improvement_threshold: float = 0.01
    n_initial_random: int = 5
    grid_points_per_param: int = 5


class AdaptiveParameterTuner:
    """
    Adaptive parameter tuning for algorithms.

    Automatically adjusts algorithm parameters to improve results
    when default parameters produce poor output.

    Example:
        tuner = AdaptiveParameterTuner()

        # Define parameter space
        space = [
            ParameterSpace("threshold", 0.1, 0.9, default=0.5),
            ParameterSpace("min_area", 100, 10000, default=1000),
        ]

        # Define scoring function
        def score_fn(params):
            result = run_algorithm(**params)
            return calculate_f1_score(result, ground_truth)

        # Run tuning
        result = tuner.tune(space, score_fn)
        print(f"Best params: {result.best_parameters}")
    """

    def __init__(self, config: Optional[ParameterTuningConfig] = None):
        """
        Initialize the parameter tuner.

        Args:
            config: Configuration options
        """
        self.config = config or ParameterTuningConfig()
        self._history: List[TuningHistory] = []

    def tune(
        self,
        parameter_space: List[ParameterSpace],
        score_function: Callable[[Dict[str, float]], float],
        baseline_score: Optional[float] = None,
    ) -> ParameterTuningResult:
        """
        Tune parameters to optimize the score function.

        Args:
            parameter_space: List of parameter definitions
            score_function: Function that takes params and returns score (higher=better)
            baseline_score: Score with default parameters (computed if None)

        Returns:
            ParameterTuningResult with best parameters found
        """
        import time
        start_time = time.time()

        # Get default parameters and baseline score
        default_params = {p.name: p.default or (p.min_value + p.max_value) / 2
                         for p in parameter_space}

        if baseline_score is None:
            baseline_score = self._evaluate(default_params, score_function)

        self._history = []
        best_params = default_params.copy()
        best_score = baseline_score

        # Run tuning based on method
        if self.config.method == TuningMethod.GRID_SEARCH:
            best_params, best_score = self._grid_search(
                parameter_space, score_function, start_time
            )
        elif self.config.method == TuningMethod.RANDOM_SEARCH:
            best_params, best_score = self._random_search(
                parameter_space, score_function, start_time
            )
        elif self.config.method == TuningMethod.ADAPTIVE:
            best_params, best_score = self._adaptive_search(
                parameter_space, score_function, best_params, best_score, start_time
            )
        else:
            # Default to random search
            best_params, best_score = self._random_search(
                parameter_space, score_function, start_time
            )

        # Calculate improvement
        improvement = best_score - baseline_score
        improvement_percent = (improvement / max(abs(baseline_score), 1e-10)) * 100

        total_duration_ms = (time.time() - start_time) * 1000

        return ParameterTuningResult(
            best_parameters=best_params,
            best_score=best_score,
            history=self._history,
            method=self.config.method,
            total_trials=len(self._history),
            total_duration_ms=total_duration_ms,
            improved=improvement > self.config.min_improvement_threshold,
            improvement_percent=improvement_percent,
            metadata={
                "baseline_score": baseline_score,
                "default_params": default_params,
            },
        )

    def tune_single_parameter(
        self,
        param: ParameterSpace,
        score_function: Callable[[float], float],
        n_points: int = 20,
    ) -> Tuple[float, float]:
        """
        Tune a single parameter.

        Args:
            param: Parameter to tune
            score_function: Function taking param value, returning score
            n_points: Number of points to evaluate

        Returns:
            Tuple of (best_value, best_score)
        """
        values = param.get_grid_values(n_points)
        best_value = param.default or values[len(values) // 2]
        best_score = float('-inf')

        for value in values:
            score = score_function(value)
            if score > best_score:
                best_score = score
                best_value = value

        return float(best_value), best_score

    def _grid_search(
        self,
        parameter_space: List[ParameterSpace],
        score_function: Callable[[Dict[str, float]], float],
        start_time: float,
    ) -> Tuple[Dict[str, float], float]:
        """Run grid search optimization."""
        import time
        from itertools import product

        # Generate grid
        param_grids = {
            p.name: p.get_grid_values(self.config.grid_points_per_param)
            for p in parameter_space
        }

        param_names = list(param_grids.keys())
        param_values = [param_grids[name] for name in param_names]

        best_params: Dict[str, float] = {}
        best_score = float('-inf')
        trials_without_improvement = 0

        for values in product(*param_values):
            # Check timeout
            elapsed = (time.time() - start_time) * 1000
            if elapsed > self.config.timeout_ms:
                break

            # Check trial limit
            if len(self._history) >= self.config.max_trials:
                break

            # Evaluate
            params = dict(zip(param_names, values))
            score = self._evaluate(params, score_function)

            if score > best_score:
                best_score = score
                best_params = params.copy()
                trials_without_improvement = 0
            else:
                trials_without_improvement += 1

            # Early stopping
            if trials_without_improvement >= self.config.early_stopping_patience:
                break

        return best_params, best_score

    def _random_search(
        self,
        parameter_space: List[ParameterSpace],
        score_function: Callable[[Dict[str, float]], float],
        start_time: float,
    ) -> Tuple[Dict[str, float], float]:
        """Run random search optimization."""
        import time

        best_params: Dict[str, float] = {}
        best_score = float('-inf')
        trials_without_improvement = 0

        for _ in range(self.config.max_trials):
            # Check timeout
            elapsed = (time.time() - start_time) * 1000
            if elapsed > self.config.timeout_ms:
                break

            # Sample random parameters
            params = {p.name: p.sample_random() for p in parameter_space}
            score = self._evaluate(params, score_function)

            if score > best_score:
                best_score = score
                best_params = params.copy()
                trials_without_improvement = 0
            else:
                trials_without_improvement += 1

            # Early stopping
            if trials_without_improvement >= self.config.early_stopping_patience:
                break

        return best_params, best_score

    def _adaptive_search(
        self,
        parameter_space: List[ParameterSpace],
        score_function: Callable[[Dict[str, float]], float],
        current_params: Dict[str, float],
        current_score: float,
        start_time: float,
    ) -> Tuple[Dict[str, float], float]:
        """Run adaptive parameter adjustment."""
        import time

        best_params = current_params.copy()
        best_score = current_score

        # Adjust one parameter at a time
        for param in parameter_space:
            # Try increasing
            test_params = best_params.copy()
            step = (param.max_value - param.min_value) * 0.1
            test_params[param.name] = min(
                param.max_value,
                best_params[param.name] + step
            )

            elapsed = (time.time() - start_time) * 1000
            if elapsed > self.config.timeout_ms:
                break

            score = self._evaluate(test_params, score_function)
            if score > best_score:
                best_score = score
                best_params = test_params.copy()

            # Try decreasing
            test_params = best_params.copy()
            test_params[param.name] = max(
                param.min_value,
                best_params[param.name] - step
            )

            elapsed = (time.time() - start_time) * 1000
            if elapsed > self.config.timeout_ms:
                break

            score = self._evaluate(test_params, score_function)
            if score > best_score:
                best_score = score
                best_params = test_params.copy()

        return best_params, best_score

    def _evaluate(
        self,
        params: Dict[str, float],
        score_function: Callable[[Dict[str, float]], float],
    ) -> float:
        """Evaluate parameters and record history."""
        import time

        start = time.time()
        try:
            score = score_function(params)
        except Exception as e:
            logger.warning(f"Evaluation failed for {params}: {e}")
            score = float('-inf')

        duration_ms = (time.time() - start) * 1000

        history = TuningHistory(
            trial_id=str(uuid.uuid4()),
            parameters=params.copy(),
            score=score,
            timestamp=datetime.now(timezone.utc),
            duration_ms=duration_ms,
        )
        self._history.append(history)

        return score
