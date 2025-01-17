"""
Computation of Persistent Diagrams
@author: Dashti Ali
"""

import numpy as np
from gudhi import CubicalComplex
import os
import time
import tqdm
import glob
import multiprocessing as mp


def compute_ph(image_path):
    image = np.load(image_path)
    max_ph_dim = 1

    if image.any():
        cub_filtration = CubicalComplex(dimensions=image.shape, top_dimensional_cells=image.flatten('F'))
        cub_filtration.persistence()
        pds = []
        
        for i in range(max_ph_dim + 1):  
            ph = cub_filtration.persistence_intervals_in_dimension(i)
            ph = ph[~np.isinf(ph).any(axis=1),:]
            pds.append(ph)
    else:
        pds = []
        print(f'{os.path.basename(image_path)} : Input data is empty!\n')

    return [os.path.basename(image_path), pds]


if __name__ == '__main__':
    start_time = time.time()
    
    main_path = 'preprocessed_liver_data'
    output_folder = 'persistent_diagrams'

    # Define datasets path
    datasets_path = {label: os.path.join(main_path, label) for label in ['ICC', 'HCC']}

    # Create the output directory
    os.makedirs(output_folder, exist_ok=True)

    for label, data_path in tqdm(datasets_path.items(), leave=False):
        args = glob.glob(os.path.join(data_path, "**", "*.npy"), recursive=True)

        pool = mp.Pool()
        results = list(tqdm(pool.imap(compute_ph, args), leave=True, total=len(args), colour='green'))
        pool.close()

        out_data = {i[0]: i[1] for i in results}
        np.save(os.path.join(output_folder, f'{label}_pds_dict.npy'), out_data)

    elapsed_time = time.time() - start_time
    print(elapsed_time)