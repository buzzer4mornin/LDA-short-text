#!/usr/bin/python
# -*- coding: utf-8 -*-

# ----- to run in terminal ------
# python ./model/run_model.py


def main():
    # -------------------------------------------- Set Config ----------------------------------------------------------
    input_folder = "./input-data"
    output_folder = "./output-data"
    saved_outputs = "./saved-outputs"
    docs_file = f"{input_folder}/docs.txt"
    setting_file = f"{input_folder}/settings.txt"

    # Create output folder if it doesn't exist
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    # -------------------------------------------- Read Data -----------------------------------------------------------
    # Read & write settings into model folder
    print('reading setting ...')
    ddict = utilities.read_setting(setting_file)

    print('write setting ...')
    saved_outputs_folder = f"{saved_outputs}/{ddict['num_topics']}_{ddict['alpha']}_{ddict['iter_infer']}_{ddict['iter_train']}/"
    if not os.path.exists(saved_outputs_folder):
        os.makedirs(saved_outputs_folder)
    file_name = f'{output_folder}/settings.txt'
    file_name_saved = f'{saved_outputs_folder}/settings.txt'
    utilities.write_setting(ddict, file_name)
    utilities.write_setting(ddict, file_name_saved)

    """
    wordids: A list whose each element is an array (word ids), corresponding to a document.
             Each element of the array is index of a unique word in the vocabulary.

    wordcts: A list whose each element is an array (word counts), corresponding to a document.
             Each element of the array says how many time the corresponding word in wordids appears
             in the document.

    E.g,
    First document = "Movie is about happy traveler"

    wordids[0] = array([127, 55, 284, 36, 47], dtype=int32)
    first document contains words whose indexes are 127th, 55th, 284th, 36th and 47th in vocabulary

    wordcts[0] = array([1, 1, 1, 1, 1], dtype=int32)
    in first document, words whose indexes are 127, 55, 284, 36, 47 appears 1, 1, 1, 1, 1 times respectively.
     """

    # Read test data
    wordids_1, wordcts_1, wordids_2, wordcts_2 = utilities.read_data_for_perpl(input_folder)

    # -------------------------------------- Initialize Algorithm --------------------------------------------------
    print('initializing LDA algorithm ...\n')
    algo = MyLDA(ddict['num_words'], ddict['num_topics'], ddict['alpha'], ddict['tau0'], ddict['kappa'], ddict['BOPE'], ddict['iter_infer'])

    # ----------------------------------------- Run Algorithm ------------------------------------------------------
    # import numpy as np
    # beta = np.load("./beta.npy")
    # list_tops = utilities.list_top(beta, ddict['tops'])
    # utilities.write_file(output_folder, list_tops, algo, 1)
    # exit()

    prev_list_tops = utilities.list_top(algo.beta, ddict['tops'])

    start = time.time()
    print('START!')
    for i in range(1, ddict['iter_train'] + 1):
        print(f'\n*** iteration: {i} ***\n')

        j = 0
        train_data = open(docs_file, 'r')
        while True:
            j += 1
            # Read mini-batch training Data
            wordids, wordcts = utilities.read_minibatch_list_frequencies(train_data, ddict['batch_size'])

            # Stop condition
            if len(wordids) == 0:
                break

            print('---num_minibatch:%d---' % (j))
            # Run single EM step
            theta = algo.run_EM(ddict['batch_size'], wordids, wordcts)

            # Compute document sparsity
            # sparsity = utilities.compute_sparsity(theta, theta.shape[0], theta.shape[1], 't')
            # print("sparsity:", sparsity)

            # Compute perplexities
            LD2_fw = utilities.compute_perplexities_fw(algo.beta, ddict['iter_infer'], wordids_1, wordcts_1,wordids_2, wordcts_2)
            LD2_vb = utilities.compute_perplexities_vb(algo.beta, ddict['alpha'], ddict['eta'], ddict['iter_infer'], wordids_1, wordcts_1, wordids_2, wordcts_2)
            print(LD2_fw)
            print(LD2_vb)

            # List Tops Difference
            # list_tops = utilities.list_top(algo.beta, ddict['tops'])
            # utilities.print_diff_list_tops(list_tops, prev_list_tops)
            # time.sleep(10)
            # prev_list_tops = list_tops

            # if i == some_iteration : --> save files
        train_data.close()
    print('DONE!')

    end = time.time()
    print("Total time spent:", end - start)

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
