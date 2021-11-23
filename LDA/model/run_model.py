#!/usr/bin/python
# -*- coding: utf-8 -*-

# ----- to run in terminal ------
# python ./model/run_model.py


def main():
    # -------------------------------------------- Set Config ----------------------------------------------------------
    docs_file = "./input-data/docs"
    setting_file = "./input-data/settings.txt"
    output_folder = "./output-data/"

    # Create model folder if it doesn't exist
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    # -------------------------------------------- Read Data -----------------------------------------------------------
    # Read & write settings into model folder
    print('reading setting ...')
    ddict = utilities.read_setting(setting_file)
    print('write setting ...')
    file_name = f'{output_folder}/settings.txt'
    utilities.write_setting(ddict, file_name)

    """
    wordids: A list whose each element is an array (words ids), corresponding to a document.
             Each element of the array is index of a unique word in the vocabulary.

    wordcts: A list whose each element is an array (word counts), corresponding to a document.
             Each element of the array says how many time the corresponding term in wordids appears
             in the document.

    E.g,
    First document = "Movie is about happy traveler"

    wordids[0] = array([127, 55, 284, 36, 47], dtype=int32)
    first document contains words whose indexes are 127th, 55th, 284th, 36th and 47th in vocabulary

    wordcts[0] = array([1, 1, 1, 1, 1], dtype=int32)
    in first document, words whose indexes are 127, 55, 284, 36, 47 appears 1, 1, 1, 1, 1 times respectively.
     """
    wordids, wordcts = utilities.read_data(docs_file)

    # -------------------------------------- Initialize Algorithm --------------------------------------------------
    print('initializing LDA algorithm ...\n')
    algo = MyLDA(ddict['num_docs'], ddict['num_terms'], ddict['num_topics'], ddict['alpha'], ddict['iter_infer'])

    # ----------------------------------------- Run Algorithm ------------------------------------------------------
    # import numpy as np
    # beta = np.load("./beta.npy")
    # list_tops = utilities.list_top(beta, ddict['tops'])
    # utilities.write_file(output_folder, list_tops, algo, 1)
    # exit()

    print('START!')
    for i in range(1, ddict['iter_train']+1):
        print(f'\n*** iteration: {i} ***\n')
        time.sleep(2)

        # Run single EM step and return attributes
        algo.run_EM(wordids, wordcts, i)

        # if i == some_iteration : --> save files

    print('DONE!')

    # ----------------------------------------- Write Results ------------------------------------------------------
    list_tops = utilities.list_top(algo.beta, ddict['tops'])
    print("\nsaving the final results.. please wait..")
    utilities.write_file(output_folder, list_tops, algo)


##########################
if __name__ == '__main__':
    import os
    import shutil
    import sys
    import time
    import pickle
    from LDA import MyLDA

    NUM_THREADS = "1"
    os.environ["OMP_NUM_THREADS"], os.environ["OPENBLAS_NUM_THREADS"], \
        os.environ["MKL_NUM_THREADS"], os.environ["VECLIB_MAXIMUM_THREADS"], \
        os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS

    sys.path.insert(0, './common')
    import utilities

    main()
