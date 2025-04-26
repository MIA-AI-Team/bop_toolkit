import pandas as pd
import numpy as np

def convert_submission(
    input_csv: str,
    method: str,
    dataset: str,
    output_csv: str = None
) -> None:
    """
    Convert a raw submission CSV into the format:
      scene_id,im_id,obj_id,score,R,t,time
    and save as METHOD_DATASET-test.csv (or output_csv if given).

    - scene_id is zero-padded to 6 digits.
    - R is flattened row‑major and written as a Python list string.
    - t is written as a Python list string.
    - time defaults to –1 if missing.
    """
    df = pd.read_csv(input_csv)

    records = []
    for _, row in df.iterrows():
        # zero‑pad scene_id to 6 digits
        scene = str(row['scene_id']).zfill(6)
        im_id = int(row['im_id'])
        obj_id = int(row['obj_id'])
        score = float(row.get('score', 1.0))

        # parse/flatten rotation matrix R
        R_raw = row['R']
        if isinstance(R_raw, str):
            R_mat = np.array(eval(R_raw))
        else:
            R_mat = np.array(R_raw).reshape(3, 3)
        R_flat = R_mat.flatten()
        R_str = ' '.join(f'{v:.16g}' for v in R_flat)

        # parse translation vector t
        t_raw = row.get('t', None)
        if t_raw is None:
            t_list = [0.0, 0.0, 0.0]
        elif isinstance(t_raw, str):
            t_list = list(eval(t_raw))
        else:
            t_list = list(t_raw)
        t_str = ' '.join(f'{v:.16g}' for v in t_list)

        # time: use provided or –1
        time_val = float(row.get('time', -1))

        records.append({
            'scene_id': scene,
            'im_id': im_id,
            'obj_id': obj_id,
            'score': score,
            'R': R_str,
            't': t_str,
            'time': time_val
        })

    out_df = pd.DataFrame(records, columns=[
        'scene_id','im_id','obj_id','score','R','t','time'
    ])

    if output_csv is None:
        output_csv = f"{method}_{dataset}-val.csv"
    out_df.to_csv(output_csv, index=False)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert a raw submission CSV into the required format for evaluation. \
                      The script parses the input CSV, processes scene_id, rotation matrices, \
                      translations, and outputs a formatted CSV.'
    )
    parser.add_argument(
        'input_csv', type=str,
        help='Path to the input CSV file containing raw submission data.'
    )
    parser.add_argument(
        'method', type=str,
        help='Method name used in generating the submission.'
    )
    parser.add_argument(
        'dataset', type=str,
        help='Dataset identifier (e.g., ipd, bop).'
    )
    parser.add_argument(
        'output_csv', type=str, nargs='?', default=None,
        help='Optional output CSV file path. If not provided, defaults to "<method>_<dataset>-val.csv".'
    )
    args = parser.parse_args()

    convert_submission(args.input_csv, args.method, args.dataset, args.output_csv)
