import io
import contextlib
import warnings
import numpy as np
import scipy as sp
from copy import deepcopy

from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.metaestimators import if_delegate_has_method

from joblib import Parallel, delayed
from hyperopt import fmin, tpe

from .utils import ParameterSampler, _check_param, _check_boosting
from .utils import _set_categorical_indexes, _get_categorical_support
from .utils import _shap_importances