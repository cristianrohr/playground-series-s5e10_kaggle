
import pandas as pd

def get_categorical_cardinality(file_path, chunksize=10000):
    """
    Calculates the cardinality of categorical columns in a large CSV file.
    """
    categorical_cols = {}
    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        if not categorical_cols:
            # First chunk, identify categorical columns
            cat_cols_in_chunk = chunk.select_dtypes(include=['object']).columns
            for col in cat_cols_in_chunk:
                categorical_cols[col] = set()

        for col in categorical_cols:
            categorical_cols[col].update(chunk[col].unique())

    cardinality = {col: len(values) for col, values in categorical_cols.items()}
    return cardinality

if __name__ == '__main__':
    file_path = 'data/fe_num_new_features/x_train.csv'
    cardinality = get_categorical_cardinality(file_path)
    print(cardinality)
