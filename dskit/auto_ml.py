"""
AutoML utilities for dskit library.
"""

import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .exceptions import DependencyError, InvalidParameterError

# Optional imports
try:
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
    from hyperopt.early_stop import no_progress_loss
    HYPEROPT_AVAILABLE = True
except ImportError:
    fmin = tpe = hp = STATUS_OK = Trials = no_progress_loss = None
    HYPEROPT_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    OPTUNA_AVAILABLE = False


def _validate_task(task):
    """Validate task is 'classification' or 'regression'."""
    task = task.lower().strip()
    if task not in ['classification', 'regression']:
        raise InvalidParameterError('task', task, ['classification', 'regression'])
    return task


def hyperopt_optimization(model_class, X, y, param_space, max_evals=50, task='classification'):
    """
    Hyperparameter optimization using Hyperopt.
    """
    if not HYPEROPT_AVAILABLE:
        raise DependencyError('hyperopt', 'Hyperopt optimization', 'pip install hyperopt')
    
    task = _validate_task(task)
    
    def objective(params):
        for key, value in params.items():
            if isinstance(value, float) and value.is_integer():
                params[key] = int(value)
        
        model = model_class(**params)
        scoring = 'accuracy' if task == 'classification' else 'r2'
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(model, X, y, cv=3, scoring=scoring)
        
        return {'loss': -scores.mean(), 'status': STATUS_OK}
    
    trials = Trials()
    best = fmin(fn=objective,
               space=param_space,
               algo=tpe.suggest,
               max_evals=max_evals,
               trials=trials,
               early_stop_fn=no_progress_loss(10))
    
    return best, trials


def optuna_optimization(model_class, X, y, param_suggestions, max_evals=50, task='classification'):
    """
    Hyperparameter optimization using Optuna.
    """
    if not OPTUNA_AVAILABLE:
        raise DependencyError('optuna', 'Optuna optimization', 'pip install optuna')
    
    task = _validate_task(task)
    
    def objective(trial):
        params = {}
        for param_name, param_config in param_suggestions.items():
            if param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
            elif param_config['type'] == 'float':
                params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'])
            elif param_config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
        
        model = model_class(**params)
        scoring = 'accuracy' if task == 'classification' else 'r2'
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(model, X, y, cv=3, scoring=scoring)
        
        return scores.mean()
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=max_evals)
    
    return study.best_params, study


def grid_search_custom(model_class, X, y, param_grid, cv=3):
    """
    Custom grid search implementation.
    """
    from sklearn.model_selection import ParameterGrid
    
    results = []
    
    for params in ParameterGrid(param_grid):
        model = model_class(**params)
        scores = cross_val_score(model, X, y, cv=cv)
        results.append({
            'params': params,
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores
        })
    
    results.sort(key=lambda x: x['mean_score'], reverse=True)
    return results


def bayesian_optimization_simple(model_class, X, y, param_bounds, max_evals=50, task='classification'):
    """
    Simple Bayesian optimization using random search with Gaussian Process.
    """
    return random_search_optimization(model_class, X, y, param_bounds, max_evals, task)


def random_search_optimization(model_class, X, y, param_bounds, max_evals=50, task='classification'):
    """
    Random search hyperparameter optimization.
    """
    import random
    
    task = _validate_task(task)
    results = []
    
    for _ in range(max_evals):
        params = {}
        for param_name, bounds in param_bounds.items():
            if bounds['type'] == 'int':
                params[param_name] = random.randint(bounds['low'], bounds['high'])
            elif bounds['type'] == 'float':
                params[param_name] = random.uniform(bounds['low'], bounds['high'])
            elif bounds['type'] == 'categorical':
                params[param_name] = random.choice(bounds['choices'])
        
        try:
            model = model_class(**params)
            scoring = 'accuracy' if task == 'classification' else 'r2'
            scores = cross_val_score(model, X, y, cv=3, scoring=scoring)
            
            results.append({
                'params': params,
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'scores': scores
            })
        except Exception as e:
            print(f"Error with params {params}: {e}")
            continue
    
    results.sort(key=lambda x: x['mean_score'], reverse=True)
    return results


def get_default_param_space(model_name, task='classification'):
    """
    Get default parameter spaces for common models.
    """
    if model_name == 'random_forest':
        return {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 200},
            'max_depth': {'type': 'int', 'low': 3, 'high': 20},
            'min_samples_split': {'type': 'int', 'low': 2, 'high': 10},
            'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 5}
        }
    elif model_name == 'xgboost':
        return {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 200},
            'max_depth': {'type': 'int', 'low': 3, 'high': 10},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3},
            'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0}
        }
    elif model_name == 'lightgbm':
        return {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 200},
            'num_leaves': {'type': 'int', 'low': 10, 'high': 100},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3},
            'feature_fraction': {'type': 'float', 'low': 0.4, 'high': 1.0}
        }
    else:
        return {'random_state': {'type': 'int', 'low': 0, 'high': 100}}


def auto_tune_model(model_class, X, y, method='random', max_evals=50, task='classification', model_name=None):
    """
    Automatically tune hyperparameters using the specified method.
    """
    if method not in ['random', 'optuna', 'hyperopt']:
        raise InvalidParameterError('method', method, ['random', 'optuna', 'hyperopt'])
    
    task = _validate_task(task)
    print(f"Starting hyperparameter tuning with {method} search...")
    
    param_space = get_default_param_space(model_name or 'random_forest', task)
    
    if method == 'random':
        results = random_search_optimization(model_class, X, y, param_space, max_evals, task)
        best_params = results[0]['params']
        best_score = results[0]['mean_score']
    
    elif method == 'optuna':
        if not OPTUNA_AVAILABLE:
            raise DependencyError('optuna', 'Optuna optimization', 'pip install optuna')
        best_params, study = optuna_optimization(model_class, X, y, param_space, max_evals, task)
        best_score = study.best_value
    
    elif method == 'hyperopt':
        if not HYPEROPT_AVAILABLE:
            raise DependencyError('hyperopt', 'Hyperopt optimization', 'pip install hyperopt')
        hyperopt_space = {}
        for param, config in param_space.items():
            if config['type'] == 'int':
                hyperopt_space[param] = hp.quniform(param, config['low'], config['high'], 1)
            elif config['type'] == 'float':
                hyperopt_space[param] = hp.uniform(param, config['low'], config['high'])
            elif config['type'] == 'categorical':
                hyperopt_space[param] = hp.choice(param, config['choices'])
        
        best_params, trials = hyperopt_optimization(model_class, X, y, hyperopt_space, max_evals, task)
        best_score = -min([trial['result']['loss'] for trial in trials.trials])
    
    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score:.4f}")
    
    best_model = model_class(**best_params)
    return best_model, best_params, best_score
