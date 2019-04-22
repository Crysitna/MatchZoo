# GPU SETTING
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.visible_device_list="1"
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

import random
import numpy as np
import pandas as pd
import keras.backend as K
from keras.optimizers import RMSprop
from keras.utils import to_categorical

import matchzoo as mz
from matchzoo.contrib.models.pwim import PWIM

from scripts.relevancy_metrics import get_map_mrr, MapMrrCallback, TensorBoardCallback
from scripts.esim_data_prep import load_train_data, load_val_data, load_test_data


# test esim with wiki_qa
epochs = 15
lstm_dim = 200
batch_size = 32
vocab_size = 6435
fixed_length_left = 32
fixed_length_right = 32

def _get_focus_mask(sim_tensor):
    """
    a copy of the _get_focus_mask function, put here for test

    sim_tensor: (T1, T2), masked
    """
    t1, t2 = K.int_shape(sim_tensor)

    t1 = 20
    t2 = 40
    sim_tensor_flattened = K.flatten(sim_tensor)      # (T1*T2)
    values, _ = tf.nn.top_k(sim_tensor_flattened, k=K.shape(sim_tensor_flattened)[-1], sorted=True) # (T1*T2)

    masks = K.zeros_like(sim_tensor)   # (T1, T2)
    for t_idx in range(t1*t2):
        value = values[t_idx]
        new_masks = K.cast(K.equal(sim_tensor, value), dtype=sim_tensor.dtype)  # (T1, T2)
        row = K.sum(new_masks, axis=1, keepdims=True)   # (T1, 1), all 0 but one 1
        col = K.sum(new_masks, axis=0, keepdims=True)   # (1, T2), all 0 but one 1
        masks = K.switch(
                    condition=K.equal(K.sum(masks * row + masks * col), 0),
                    then_expression= 0.9 * new_masks + masks,
                    else_expression=masks
                )
    masks += 0.1 * K.ones_like(sim_tensor)
    return masks

# construct model
def prepare_model(load_emb=False, preprocessor=None):
    model = PWIM()
    classification_task = mz.tasks.Classification(num_classes=2)
    model.params['task'] = classification_task
    model.params['mask_value'] = 0
    model.params['input_shapes'] = [[fixed_length_left, ],
                                    [fixed_length_right, ]]
    model.params['lstm_dim'] = lstm_dim
    # model.params['embedding_input_dim'] = preprocessor.context['vocab_size']
    model.params['embedding_input_dim'] = vocab_size
    model.params['embedding_output_dim'] = 300
    model.params['embedding_trainable'] = False

    model.params['mlp_num_units'] = 128
    model.params['mlp_num_layers'] = 0
    model.params['mlp_num_fan_out'] = 128
    model.params['mlp_activation_func'] = 'relu'
    model.params['optimizer'] = RMSprop(lr=1e-4)

    model.guess_and_fill_missing_params()
    model.build()

    model.compile(loss='hinge')
    model.backend.summary() # not visualize

    if load_emb:
        assert (preprocessor != None)
        import csv
        file_path = "~/datasets/pretrain_emb/glove.840B.300d.txt"
        data = pd.read_table(file_path,
                             sep=" ",
                             index_col=0,
                             header=None,
                             quoting=csv.QUOTE_NONE)

        googlenews_embedding = mz.embedding.Embedding(data)
        embedding_matrix = googlenews_embedding.build_matrix(preprocessor.context['vocab_unit'].state['term_index'], initializer=lambda: 0)
        model.load_embedding_matrix(embedding_matrix)

    return model

def _print_pad(arr, name):
    print(name, (arr == 0).sum(axis=-1))

def _print_pad_matrix(arr, name):
    """ print out number of pad in both dimension"""
    arr = (arr == 0)
    print(name+" axis 0: \n", arr.sum(axis=-1))    # (xx, a, 1)
    print(name+" axis 1: \n", arr.sum(axis=-2))    # (xx, 1, b)

def prepare_sample_input(rand=True, num_pad=None):
    if rand:
        X_left = np.random.randint(low=1, high=vocab_size,
                                   size=(batch_size, fixed_length_left))
        X_right = np.random.randint(low=1, high=vocab_size,
                                    size=(batch_size, fixed_length_right))
    else:
        X_left = [ list(range(10)) + list(range(10)) for _ in range(batch_size)]
        X_right = [list(range(20)) + list(range(20)) for _ in range(batch_size)]

    Y = np.random.shuffle([(0, 1) for _ in range(5)] + [(1, 0) for _ in range(5)])

    pads_left = []
    pads_right = []
    for i in range(batch_size):
        if num_pad == None:
            num_pad = random.randint(0, 10)
        pads_left.append(num_pad)
        X_left[i, -num_pad:] = 0

        if num_pad == None:
            num_pad = random.randint(0, 15)
        X_right[i, -num_pad:] = 0

        pads_right.append(num_pad)

    _print_pad(X_left, name="X_left")
    _print_pad(X_right, name="X_right")

    mask = np.array([np.dot(l[:, None], r[None, :]) != 0 for l, r in zip(X_left, X_right)], dtype=np.float64)
    return X_left, X_right, Y, mask

def prepare_true_input():
    preprocessor = mz.preprocessors.BasicPreprocessor(fixed_length_left=fixed_length_left,
                                                      fixed_length_right=fixed_length_right,
                                                      remove_stop_words=False,
                                                      filter_low_freq=10)

    train_X, train_Y_, preprocessor = load_train_data(preprocessor, False)
    for k in train_X.keys():
        train_X[k] = train_X[k][:batch_size]
    Y = to_categorical(train_Y_)[:batch_size]
    return train_X, Y, preprocessor

def test_mask_zero():
    """
    assume model set [h1_for, h1_back, h2_for, h2_back] as output
    """
    X_left, X_right, Y, _ = prepare_sample_input()
    model = prepare_model()

    # check if the "0" position is not calculated
    hs = model.predict(x=[X_left, X_right], batch_size=batch_size)
    for i in range(batch_size):
        print(i, "the batch")
        for n, h in zip(["for:\t", "back:\t"], hs[:2]):
            pad_l = pads_left[i]
            pad_r = pads_right[i]
            print(n, h.sum(-1)[i, -(pad_l+1):])

        for n, h in zip(["for:\t", "back:\t"], hs[2:]):
            pad_l = pads_left[i]
            pad_r = pads_right[i]
            print(n, h.sum(-1)[i, -(pad_l+1):])
        print()

def test_sim_cube():
    X_left, X_right, Y, mask = prepare_sample_input(rand=True)
    model = prepare_model()

    sim_cube = model.predict(x=[X_left, X_right], batch_size=batch_size)
    print("sim_cube: shape ", sim_cube.shape)   # (5, 13, 20, 40)

    # check if sim_cube is masked correctly
    s = (sim_cube[:, 1, :, :] == 0)     # (B, T1, T2)
    print("along S1: ", s.sum(axis=1))
    print("along S2: ", s.sum(axis=2))
    print()
    return sim_cube, mask

def test_get_focus_mask():
    # test with previous models
    # sim_cube, h12_mask = test_sim_cube()
    # sim_tensor = tf.convert_to_tensor(sim_cube[:, 10, :, :])

    # synthetic data only for focus mask
    # sim_tensor_n = np.reshape(list(range(fixed_length_left*fixed_length_right)[::-1]),
    #                           (fixed_length_left, fixed_length_right))
    sim_tensor_n = np.random.rand(fixed_length_left, fixed_length_right)
    sim_tensor = tf.convert_to_tensor(
        np.array([sim_tensor_n for _ in range(batch_size)]),
        dtype=tf.float64)
    masks = K.map_fn(_get_focus_mask, sim_tensor)
    masks = K.eval(masks)
    # masks = K.eval(masks) * h12_mask
    for m in masks: # (T1, T2), check is it no 1 is given at any pad places
        print("*" * 10)
        sorted_idx = np.argsort(m, axis=None)

        x_collection = []
        y_collection = []
        for idx in sorted_idx[::-1]:
            x_i = idx // 32
            y_i = idx % 32
            if(m[x_i][y_i] != 1 and (not (m[x_i][y_i] > 0.0999 and m[x_i][y_i]) < 0.1001)):
                print("exceptions: ", x_i, y_i, m[x_i][y_i])

            if(m[x_i][y_i] > 0.11):
                print(x_i, y_i,"\torigin: ", sim_tensor_n[x_i][y_i], "\tmask value:", m[x_i][y_i])
                x_collection.append(x_i)
                y_collection.append(y_i)
        print("num repetitive in x: ", len(set(x_collection)) - len(x_collection))
        print("num repetitive in y: ", len(set(y_collection)) - len(y_collection))

def test_focuscube_layer_pad():
    X_left, X_right, Y, _ = prepare_sample_input()
    model = prepare_model()

    focuscubes = model.predict(x=[X_left, X_right], batch_size=batch_size)
    for i in range(batch_size):
        print(i, "the batch")
        focuscube = focuscubes[i, :, :, 0]
        _print_pad_matrix(focuscube, "{}-th".format(i))
        print()

def test_focuscube_layer_mask():
    X_left, X_right, Y, _ = prepare_sample_input()
    model = prepare_model()

    focuscubes = model.predict(x=[X_left, X_right], batch_size=batch_size)
    for i in range(batch_size):
        print(i, "the batch")

def test_focuscube_single_layer_content():
    # X_left, X_right, Y, _ = prepare_sample_input(num_pad=28)
    X_left, X_right, Y, _ = prepare_true_input()
    model = prepare_model()

    sims, focuscubes = model.predict(x=[X_left, X_right], batch_size=batch_size)
    print("=="*20, "FOCUS CUBE: ", focuscubes.shape)

    for i in range(batch_size):
        print(i, "the batch")
        focuscube = focuscubes[i, 0, :, :]
        # focuscube = focuscubes[i, :, :]
        sim = sims[i, 10, :, :]

        sim_s = np.sort(sim, axis=None)
        # print(sim_s[-10:])
        # print(sim_s[sim_s>-1e5][:10])


        # look into the simcube:
        print("cos")
        sim_cos = np.sort(sims[i, 10, :, :], axis=None)
        print("largest few: \n", sim_cos[-10:][::-1])
        print("smallest few: \n", sim_cos[sim_cos>-1e5][:10])

        print("l2")
        sim_l2 = np.sort(sims[i, 11, :, :], axis=None)
        print("largest few: \n", sim_l2[-10:][::-1])
        print("smallest few: \n", sim_l2[sim_l2>-1e5][:10])

        print("dot")
        sim_dot = np.sort(sims[i, 12, :, :], axis=None)
        print("largest few: \n", sim_dot[-10:][::-1])
        print("smallest few: \n", sim_dot[sim_dot>-1e5][:10])

        sorted_idx = np.argsort(focuscube, axis=None)

        for idx in sorted_idx[::-1]:
            x_i = idx // 32
            y_i = idx % 32
            if focuscube[x_i][y_i] > 0.11:
            # if True:
                print("[{}][{}]: {}".format(x_i, y_i, focuscube[x_i][y_i]), end="\t")
                print("cos: ", sims[i, 10, :, :][x_i][y_i], end="\t")
                print("l2: ", sims[i, 11, :, :][x_i][y_i], end="\t")
                print("dot: ", sims[i, 12, :, :][x_i][y_i])

def test_focuscube_visualize():
    X, Y, preprocessor = prepare_true_input()
    X_left, X_right = X['text_left'], X['text_right']
    model = prepare_model()
    sims, focuses = model.predict(x=[X_left, X_right], batch_size=1)

    s = sims[3]
    f = focuses[3]

    import seaborn as sns
    import matplotlib.pyplot as plt

    names = ["pad", 'bi_cos', 'bi_l2', 'bi_dot', 'for_cos', 'for_l2', 'for_dot', 'back_cos', 'back_l2', 'back_dot', 'add_cos', 'add_l2', 'add_dot']
    for i, name in enumerate(names):
        fig = plt.figure()
        plt.subplot(2, 1, 1)
        s[i, :, :][s[i, :, :] < -1e4] = s[i, :, :][s[i, :, :] > -1e4].min() - 0.01
        sns.heatmap(s[i, :, :])
        plt.subplot(2, 1, 2)
        # f[:, :, i][f[:, :, i] < -1e4] = f[:, :, i][f[:, :, i] > -1e4].min() - 0.01
        sns.heatmap(f[:, :, i])
        fig.suptitle(name)
        plt.savefig("{}.png".format(i))

def inspect_output():
    X, Y, preprocessor = prepare_true_input()
    X_left, X_right = X['text_left'], X['text_right']
    model = prepare_model()

    preds = model.predict(x=[X_left, X_right], batch_size=batch_size)
    
    for t, p in zip(Y, preds):
        print("true: {}\t pred: {}".format(t, p))

def mini_train():
    X, Y, preprocessor = prepare_true_input()
    model = prepare_model(load_emb=True, preprocessor=preprocessor)

    history = model.fit(x = [X['text_left'],
                             X['text_right']],
                        y = Y,                                  # (20360, 2)
                        validation_data = (X, Y),
                        batch_size = batch_size,
                        epochs = epochs,
                        callbacks=[MapMrrCallback(X, Y), 
                                   TensorBoardCallback(logdir='./logdir', update_freq=1e3)])
    preds = model.predict(x=[X_left, X_right], batch_size=batch_size)

# test_focuscube_visualize()
mini_train()