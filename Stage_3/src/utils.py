import polars as pl
import sys
import numpy as np

try:
    from numba import jit, int32, boolean
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components as scipy_cc
except:
    print("Numba not installed. Some functions will not work.")

try:
    import cugraph
except:
    print("cuGraph not installed. Some functions will not work.")


def connected_components(adj: csr_matrix):
    print("Finding connected components...")
    print(f"Number of nodes: {adj.shape[0]}")
    print(f"Number of edges: {adj.count_nonzero()}")

    try:
        print("Using cuGraph...")
        # Convert adj from bool to int32
        result = cugraph.strongly_connected_components(
            adj.astype("int32"), directed=False, return_labels=True
        )
        print("cuGraph finished.")
        print(result)
        return result
    except Exception as e:
        print(f"We encountered an error while using cugraph: {e}")

    try:
        n = adj.shape[0]
        visited = np.zeros(n, dtype=np.bool_)
        labels = np.zeros(n, dtype=np.int32)
        current_label = 0

        @jit(nopython=True)
        def dfs(node, current_label, labels, visited, data, indices, indptr):
            visited[node] = True
            labels[node] = current_label
            for i in range(indptr[node], indptr[node + 1]):
                neighbor = indices[i]
                if data[i] and not visited[neighbor]:
                    dfs(neighbor, current_label, labels, visited, data, indices, indptr)

        data = adj.data.astype(np.bool_)
        indices = adj.indices
        indptr = adj.indptr

        for i in range(n):
            if not visited[i]:
                current_label += 1
                dfs(i, current_label, labels, visited, data, indices, indptr)

        return current_label, labels
    except Exception as e:
        print(f"We encountered an error while using numba: {e}")
        return scipy_cc(adj.astype(np.int32), directed=False, return_labels=True)


def select_first_in_group(df, connected_component_col, selected_col, index_col="index"):
    # Group by connected_component_col and select the first row of each group
    first_rows = (
        df.sort(connected_component_col)
        .group_by(connected_component_col, maintain_order=True)
        .agg(pl.col(index_col).first())
    )

    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    # Create a set of the index values of the first rows
    first_row_indices = set(first_rows[index_col].to_list())

    # Add a new column selected_col to the original dataframe,
    # which is True if the row index is in first_row_indices and False otherwise
    df = df.with_columns(
        pl.when(pl.col(index_col).is_in(first_row_indices))
        .then(True)
        .otherwise(False)
        .alias(selected_col)
    )

    # Return the modified dataframe
    return df


def load_pl_dataframe(path, lazy=False):
    """Load a Polars DataFrame from a file.
    Args:
        path (str): Path to the file.
    Returns:
        polars.DataFrame: The loaded DataFrame.
    """
    import polars as pl

    if path.endswith(".parquet"):
        if lazy:
            return pl.scan_parquet(path)
        else:
            return pl.read_parquet(path)
    elif path.endswith(".csv"):
        if lazy:
            return pl.scan_csv(path)
        else:
            return pl.read_csv(path)
    else:
        raise ValueError(f"Unknown file type: {path}")


def write_pl_dataframe(df, path, lazy=False):
    """Write a Polars DataFrame to a file.
    Args:
        df (polars.DataFrame): The DataFrame to write.
        path (str): Path to the file.
    """
    if path.endswith(".parquet"):
        if lazy:
            df.sink_parquet(path)
        else:
            df.write_parquet(path)
    elif path.endswith(".csv"):
        if lazy:
            df.sink_csv(path)
        else:
            df.write_csv(path)
    else:
        raise ValueError(f"Unknown file type: {path}")


def flatten_struct_column(df, column_name):
    """
    Flatten a struct column in a DataFrame into multiple columns.

    This function takes a DataFrame and a struct column name, and flattens
    the struct into multiple columns, with each field in the struct becoming
    a separate column in the DataFrame.

    Parameters:
    df (DataFrame): The DataFrame containing the struct column.
    column_name (str): The name of the struct column to be flattened.

    Returns:
    DataFrame: A new DataFrame with the struct column flattened into multiple columns.
    """
    fields = df.schema[column_name].fields
    select_expressions = [
        pl.col(column_name)
        .struct.field(field.name)
        .alias(f"{column_name}_{field.name}")
        for field in fields
    ]
    df = df.with_columns(select_expressions)
    return df


def calculate_struct_column_differences(df, col1, col2, new_struct_name):
    """
    Calculate the differences between two struct columns and create a new struct column.

    This function flattens two struct columns, calculates the differences between
    corresponding fields, and creates a new struct column with these differences.

    Parameters:
    df (DataFrame): The DataFrame containing the struct columns.
    col1 (str): The name of the first struct column.
    col2 (str): The name of the second struct column.
    new_struct_name (str): The name for the new struct column containing the differences.

    Returns:
    DataFrame: A new DataFrame with the new struct column added.
    """
    df = flatten_struct_column(df, col1)
    df = flatten_struct_column(df, col2)

    difference_expressions = []
    to_drop = []
    for field in df.schema[col1].fields:
        field_name = field.name
        diff_expr = (
            pl.col(f"{col1}_{field_name}") - pl.col(f"{col2}_{field_name}")
        ).alias(field_name)
        to_drop.append(f"{col1}_{field_name}")
        to_drop.append(f"{col2}_{field_name}")
        difference_expressions.append(diff_expr)

    df = df.with_columns(pl.struct(difference_expressions).alias(new_struct_name)).drop(
        to_drop
    )
    return df


def calculate_struct_column_minimums(df, col1, col2, new_struct_name):
    """
    Calculate the minimum values between two struct columns and create a new struct column.

    This function flattens two struct columns, calculates the minimum values between
    corresponding fields, and creates a new struct column with these minimum values.

    Parameters:
    df (DataFrame): The DataFrame containing the struct columns.
    col1 (str): The name of the first struct column.
    col2 (str): The name of the second struct column.
    new_struct_name (str): The name for the new struct column containing the minimum values.

    Returns:
    DataFrame: A new DataFrame with the new struct column added.
    """
    df = flatten_struct_column(df, col1)
    df = flatten_struct_column(df, col2)

    minimum_expressions = []
    to_drop = []
    for field in df.schema[col1].fields:
        field_name = field.name
        min_expr = (
            pl.min_horizontal(f"{col1}_{field_name}", f"{col2}_{field_name}")
        ).alias(field_name)
        to_drop.append(f"{col1}_{field_name}")
        to_drop.append(f"{col2}_{field_name}")
        minimum_expressions.append(min_expr)

    df = df.with_columns(pl.struct(minimum_expressions).alias(new_struct_name)).drop(
        to_drop
    )
    return df
