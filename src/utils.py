import numpy as np
import pandas as pd

VALID_X_TYPES = (pd.DataFrame, np.ndarray)  # nested pd.DataFrame, 2d or 3d np.array
VALID_Y_TYPES = (pd.Series, np.ndarray)  # 1-d vector


def check_X(
    X,
    enforce_univariate=False,
    enforce_min_instances=1,
    enforce_min_columns=1,
    coerce_to_numpy=False,
):
    """Validate input data.
    Parameters
    ----------
    X : pd.DataFrame or np.array
        Input data
    enforce_univariate : bool, optional (default=False)
        Enforce that X is univariate.
    enforce_min_instances : int, optional (default=1)
        Enforce minimum number of instances.
    enforce_min_columns : int, optional (default=1)
        Enforce minimum number of columns (or time-series variables).
    coerce_to_numpy : bool, optional (default=False)
        If True, X will be coerced to a 3-dimensional numpy array.

    Returns
    -------
    X : pd.DataFrame or np.array
        Checked and possibly converted input data
    Raises
    ------
    ValueError
        If X is invalid input data
    """

    if not isinstance(X, VALID_X_TYPES):
        raise ValueError(
            f"X must be a pd.DataFrame or a np.array, " f"but found: {type(X)}"
        )

    # check np.array
    # check first if we have the right number of dimensions, otherwise we
    # may not be able to get the shape of the second dimension below
    if isinstance(X, np.ndarray):
        if X.ndim == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        elif X.ndim == 1 or X.ndim > 3:
            raise ValueError(
                f"If passed as a np.array, X must be a 2 or 3-dimensional "
                f"array, but found shape: {X.shape}"
            )

    # enforce minimum number of columns
    n_columns = X.shape[1]
    if n_columns < enforce_min_columns:
        raise ValueError(
            f"X must contain at least: {enforce_min_columns} columns, "
            f"but found only: {n_columns}."
        )

    # enforce univariate data
    if enforce_univariate and n_columns > 1:
        raise ValueError(
            f"X must be univariate with X.shape[1] == 1, but found: "
            f"X.shape[1] == {n_columns}."
        )

    # enforce minimum number of instances
    if enforce_min_instances > 0:
        _enforce_min_instances(X, min_instances=enforce_min_instances)

    # check pd.DataFrame
    if isinstance(X, pd.DataFrame):
        if not is_nested_dataframe(X):
            raise ValueError(
                "If passed as a pd.DataFrame, X must be a nested "
                "pd.DataFrame, with pd.Series or np.arrays inside cells."
            )
        # convert pd.DataFrame
        if coerce_to_numpy:
            X = from_nested_to_3d_numpy(X)

    return X


def _enforce_min_instances(x, min_instances=1):
    n_instances = x.shape[0]
    if n_instances < min_instances:
        raise ValueError(
            f"Found array with: {n_instances} instance(s) "
            f"but a minimum of: {min_instances} is required."
        )


def from_nested_to_3d_numpy(X):
    """Convert nested Panel to 3D numpy Panel.
    Convert nested pandas DataFrame (with time series as pandas Series
    in cells) into NumPy ndarray with shape
    (n_instances, n_columns, n_timepoints).
    Parameters
    ----------
    X : pd.DataFrame
        Nested pandas DataFrame
    Returns
    -------
    X_3d : np.ndarrray
        3-dimensional NumPy array
    """
    if not is_nested_dataframe(X):
        raise ValueError("Input DataFrame is not a nested DataFrame")

    # n_columns = X.shape[1]
    nested_col_mask = [*are_columns_nested(X)]

    # If all the columns are nested in structure
    if nested_col_mask.count(True) == len(nested_col_mask):
        X_3d = np.stack(
            X.applymap(_convert_series_cell_to_numpy)
            .apply(lambda row: np.stack(row), axis=1)
            .to_numpy()
        )

    # If some columns are primitive (non-nested) then first convert to
    # multi-indexed DataFrame where the same value of these columns is
    # repeated for each timepoint
    # Then the multi-indexed DataFrame can be converted to 3d NumPy array
    else:
        X_mi = from_nested_to_multi_index(X)
        X_3d = from_multi_index_to_3d_numpy(
            X_mi, instance_index="instance", time_index="timepoints"
        )

    return X_3d


def _convert_series_cell_to_numpy(cell):
    if isinstance(cell, pd.Series):
        return cell.to_numpy()
    else:
        return cell


def _cell_is_series_or_array(cell):
    return isinstance(cell, (pd.Series, np.ndarray))


def _nested_cell_mask(X):
    return X.applymap(_cell_is_series_or_array)


def are_columns_nested(X):
    """Check whether any cells have nested structure in each DataFrame column.
    Parameters
    ----------
    X : pd.DataFrame
        DataFrame to check for nested data structures.
    Returns
    -------
    any_nested : bool
        If True, at least one column is nested.
        If False, no nested columns.
    """
    any_nested = _nested_cell_mask(X).any().values
    return any_nested


def is_nested_dataframe(obj, return_metadata=False, var_name="obj"):
    """Check whether the input is a nested DataFrame.
    To allow for a mixture of nested and primitive columns types the
    the considers whether any column is a nested np.ndarray or pd.Series.
    Column is consider nested if any cells in column have a nested structure.
    Parameters
    ----------
    X: Input that is checked to determine if it is a nested DataFrame.
    Returns
    -------
    bool: Whether the input is a nested DataFrame
    """
    # If not a DataFrame we know is_nested_dataframe is False
    if not isinstance(obj, pd.DataFrame):
        msg = f"{var_name} must be a pd.DataFrame, found {type(obj)}"
        return _ret(False, msg, None, return_metadata)

    # Otherwise we'll see if any column has a nested structure in first row
    else:
        if not are_columns_nested(obj).any():
            msg = f"{var_name} entries must be pd.Series"
            return _ret(False, msg, None, return_metadata)

    metadata = dict()
    metadata["is_univariate"] = obj.shape[1] < 2
    metadata["n_instances"] = len(obj)
    metadata["is_one_series"] = len(obj) == 1
    if return_metadata:
        metadata["has_nans"] = _nested_dataframe_has_nans(obj)
        metadata["is_equal_length"] = not _nested_dataframe_has_unequal(obj)

    # todo: this is temporary override, proper is_empty logic needs to be added
    metadata["is_empty"] = False
    metadata["is_equally_spaced"] = True
    # end hacks

    return _ret(True, None, metadata, return_metadata)


def _nested_dataframe_has_nans(X: pd.DataFrame) -> bool:
    """Check whether an input pandas of Series has nans.
    Parameters
    ----------
    X : pd.DataFrame where each cell is a pd.Series
    Returns
    -------
    True if x contains any NaNs, False otherwise.
    """
    cases = len(X)
    dimensions = len(X.columns)
    for i in range(cases):
        for j in range(dimensions):
            s = X.iloc[i, j]
            for k in range(s.size):
                if pd.isna(s.iloc[k]):
                    return True
    return False


def _nested_dataframe_has_unequal(X: pd.DataFrame) -> bool:
    """Check whether an input nested DataFrame of Series has unequal length series.
    Parameters
    ----------
    X : pd.DataFrame where each cell is a pd.Series
    Returns
    -------
    True if x contains any NaNs, False otherwise.
    """
    rows = len(X)
    cols = len(X.columns)
    s = X.iloc[0, 0]
    length = len(s)

    for i in range(0, rows):
        for j in range(0, cols):
            s = X.iloc[i, j]
            temp = len(s)
            if temp != length:
                return True
    return False


def _ret(valid, msg, metadata, return_metadata):
    if return_metadata:
        return valid, msg, metadata
    else:
        return valid


def from_nested_to_multi_index(X, instance_index=None, time_index=None):
    """Convert nested pandas Panel to multi-index pandas Panel.
    Converts nested pandas DataFrame (with time series as pandas Series
    or NumPy array in cells) into multi-indexed pandas DataFrame.
    Can convert mixed nested and primitive DataFrame to multi-index DataFrame.
    Parameters
    ----------
    X : pd.DataFrame
        The nested DataFrame to convert to a multi-indexed pandas DataFrame
    instance_index : str
        Name of the multi-index level corresponding to the DataFrame's instances
    time_index : str
        Name of multi-index level corresponding to DataFrame's timepoints
    Returns
    -------
    X_mi : pd.DataFrame
        The multi-indexed pandas DataFrame
    """
    if not is_nested_dataframe(X):
        raise ValueError("Input DataFrame is not a nested DataFrame")

    if time_index is None:
        time_index_name = "timepoints"
    else:
        time_index_name = time_index

    # n_columns = X.shape[1]
    nested_col_mask = [*are_columns_nested(X)]

    if instance_index is None:
        instance_idxs = X.index.get_level_values(-1).unique()
        # n_instances = instance_idxs.shape[0]
        instance_index_name = "instance"

    else:
        if instance_index in X.index.names:
            instance_idxs = X.index.get_level_values(instance_index).unique()
        else:
            instance_idxs = X.index.get_level_values(-1).unique()
        # n_instances = instance_idxs.shape[0]
        instance_index_name = instance_index

    instances = []
    for instance_idx in instance_idxs:
        instance = [
            _val if isinstance(_val, pd.Series) else pd.Series(_val, name=_lab)
            for _lab, _val in X.loc[instance_idx, :].iteritems()  # noqa
        ]

        instance = pd.concat(instance, axis=1)
        # For primitive (non-nested column) assume the same
        # primitive value applies to every timepoint of the instance
        for col_idx, is_nested in enumerate(nested_col_mask):
            if not is_nested:
                instance.iloc[:, col_idx] = instance.iloc[:, col_idx].ffill()

        # Correctly assign multi-index
        multi_index = pd.MultiIndex.from_product(
            [[instance_idx], instance.index],
            names=[instance_index_name, time_index_name],
        )
        instance.index = multi_index
        instances.append(instance)

    X_mi = pd.concat(instances)
    X_mi.columns = X.columns

    return X_mi


def from_multi_index_to_3d_numpy(X, instance_index=None, time_index=None):
    """Convert pandas multi-index Panel to numpy 3D Panel.
    Convert panel data stored as pandas multi-index DataFrame to
    Numpy 3-dimensional NumPy array (n_instances, n_columns, n_timepoints).
    Parameters
    ----------
    X : pd.DataFrame
        The multi-index pandas DataFrame
    instance_index : str
        Name of the multi-index level corresponding to the DataFrame's instances
    time_index : str
        Name of multi-index level corresponding to DataFrame's timepoints
    Returns
    -------
    X_3d : np.ndarray
        3-dimensional NumPy array (n_instances, n_columns, n_timepoints)
    """
    if X.index.nlevels != 2:
        raise ValueError("Multi-index DataFrame should have 2 levels.")

    if (instance_index is None) or (time_index is None):
        msg = "Must supply parameters instance_index and time_index"
        raise ValueError(msg)

    n_instances = len(X.groupby(level=instance_index))

    n_timepoints = len(X.groupby(level=time_index))

    n_columns = X.shape[1]

    X_3d = X.values.reshape(n_instances, n_timepoints, n_columns).swapaxes(1, 2)

    return X_3d
