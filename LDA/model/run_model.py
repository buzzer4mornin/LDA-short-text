#!/usr/bin/python
# -*- coding: utf-8 -*-

# ----- to run in terminal ------
# python ./model/run_model.py


def main():
    # -------------------------------------------- Set Config ----------------------------------------------------------
    docs_file = "./input-data/docs.txt"
    setting_file = "./input-data/settings.txt"
    output_folder = "./showcase_50_50/"
    saved_outputs = "./saved-outputs/"

    # Create output folder if it doesn't exist
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    # -------------------------------------------- Read Data -----------------------------------------------------------
    # Read & write settings into model folder
    print('reading setting ...')
    ddict = utilities.read_setting(setting_file)
    saved_outputs_folder = f"{saved_outputs}/{ddict['num_topics']}_{ddict['alpha']}_{ddict['iter_infer']}_{ddict['iter_train']}/"
    try:
        os.makedirs(saved_outputs_folder)
    except:
        pass

    print('write setting ...')
    file_name = f'{output_folder}/settings.txt'
    file_name_saved = f'{saved_outputs_folder}/settings.txt'
    utilities.write_setting(ddict, file_name)
    utilities.write_setting(ddict, file_name_saved)

    """
    termids: A list whose each element is an array (terms ids), corresponding to a document.
             Each element of the array is index of a unique term in the vocabulary.

    termcts: A list whose each element is an array (term counts), corresponding to a document.
             Each element of the array says how many time the corresponding term in termids appears
             in the document.

    E.g,
    First document = "Movie is about happy traveler"

    termids[0] = array([127, 55, 284, 36, 47], dtype=int32)
    first document contains terms whose indexes are 127th, 55th, 284th, 36th and 47th in vocabulary

    termcts[0] = array([1, 1, 1, 1, 1], dtype=int32)
    in first document, terms whose indexes are 127, 55, 284, 36, 47 appears 1, 1, 1, 1, 1 times respectively.
     """
    termids, termcts = utilities.read_data(docs_file)

    # -------------------------------------- Initialize Algorithm --------------------------------------------------
    print('initializing LDA algorithm ...\n')
    algo = MyLDA(ddict['num_docs'], ddict['num_terms'], ddict['num_topics'], ddict['alpha'], ddict['iter_infer'])

    # ----------------------------------------- Run Algorithm ------------------------------------------------------
    # import numpy as np
    # beta = np.load("./beta.npy")
    # list_tops = utilities.list_top(beta, ddict['tops'])
    # utilities.write_file(output_folder, list_tops, algo, 1)
    # exit()

    prev_list_tops = utilities.list_top(algo.beta, ddict['tops'])

    print('START!')
    for i in range(1, ddict['iter_train'] + 1):
        print(f'\n*** iteration: {i} ***\n')
        time.sleep(2)

        # Run single EM step and return attributes
        algo.run_EM(termids, termcts, i)

        # List Tops Difference
        list_tops = utilities.list_top(algo.beta, ddict['tops'])
        utilities.print_diff_list_tops(list_tops, prev_list_tops)
        time.sleep(10)
        prev_list_tops = list_tops

        # if i == some_iteration : --> save files

    print('DONE!')

    # ----------------------------------------- Write Results ------------------------------------------------------
    list_tops = utilities.list_top(algo.beta, ddict['tops'])
    print("\nsaving the final results.. please wait..")
    utilities.write_file(output_folder, saved_outputs_folder, list_tops, algo)


if __name__ == '__main__':
    import os
    import shutil
    import sys
    import time
    import pickle
    from LDA import MyLDA

    NUM_THREADS = "1"
    os.environ["OMP_NUM_THREADS"] = NUM_THREADS
    os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS
    os.environ["MKL_NUM_THREADS"] = NUM_THREADS
    os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS
    os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS

    sys.path.insert(0, './common')
    import utilities

    main()
