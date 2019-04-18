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
from keras.optimizers import Adam
from keras.utils import to_categorical

import matchzoo as mz
from matchzoo.contrib.models.pwim import PWIM

from scripts.relevancy_metrics import get_map_mrr, MapMrrCallback, TensorBoardCallback
from scripts.esim_data_prep import load_train_data, load_val_data, load_test_data


# test esim with wiki_qa
epochs = 15
lstm_dim = 200
batch_size = 5
vocab_size = 3000
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
def prepare_model():
    model = PWIM()
    classification_task = mz.tasks.Classification(num_classes=2)
    model.params['task'] = classification_task
    model.params['mask_value'] = 0
    model.params['input_shapes'] = [[fixed_length_left, ],
                                    [fixed_length_right, ]]
    model.params['lstm_dim'] = lstm_dim
    # model.params['embedding_input_dim'] = preprocessor.context['vocab_size']
    model.params['embedding_input_dim'] = vocab_size
    model.params['embedding_output_dim'] = lstm_dim
    model.params['embedding_trainable'] = False

    model.params['mlp_num_units'] = 128
    model.params['mlp_num_layers'] = 0
    model.params['mlp_num_fan_out'] = 128
    model.params['mlp_activation_func'] = 'relu'
    model.params['optimizer'] = Adam(lr=1e-4)

    model.guess_and_fill_missing_params()
    model.build()

    model.compile(loss='hinge')
    model.backend.summary() # not visualize

    return model

def _print_pad(arr, name):
    print(name, (arr == 0).sum(axis=-1))

def _print_pad_matrix(arr, name):
    """ print out number of pad in both dimension"""
    arr = (arr == 0)
    print(name+" axis 0: \n", arr.sum(axis=-1))    # (xx, a, 1)
    print(name+" axis 1: \n", arr.sum(axis=-2))    # (xx, 1, b)

def prepare_sample_input(rand=True):
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
        num_pad = random.randint(0, 10)
        pads_left.append(num_pad)
        X_left[i, -num_pad:] = 0

        num_pad = random.randint(0, 15)
        X_right[i, -num_pad:] = 0
        pads_right.append(num_pad)

    _print_pad(X_left, name="X_left")
    _print_pad(X_right, name="X_right")

    mask = np.array([np.dot(l[:, None], r[None, :]) != 0 for l, r in zip(X_left, X_right)], dtype=np.float64)
    return X_left, X_right, Y, mask

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
            # assert (m[x_i][y_i] == 1 or m[x_i][y_i] == 0.1)
            if(m[x_i][y_i] != 1 and (not (m[x_i][y_i] > 0.0999 and m[x_i][y_i]) < 0.1001)):
                print("exceptions: ", x_i, y_i, m[x_i][y_i])

            # if(m[x_i][y_i] > 0.999 and m[x_i][y_i] < 1.001):
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


def test_focuscube_layer_content():
    X_left, X_right, Y, _ = prepare_sample_input()
    model = prepare_model()

    sims, focuscubes = model.predict(x=[X_left, X_right], batch_size=batch_size)
    for i in range(batch_size):
        print(i, "the batch")
        # focuscube = focuscubes[i, :, :, 0]
        focuscube = focuscubes[i, :, :]
        sim = sims[i, 11, :, :]

        sim_s = np.sort(sim, axis=None)[-10:]
        print(sim_s)
        sorted_idx = np.argsort(focuscube, axis=None)

        for idx in sorted_idx[::-1]:
            x_i = idx // 32
            y_i = idx % 32
            if focuscube[x_i][y_i] > 0.11:
                print("[{}][{}]: {} {}".format(x_i, y_i, focuscube[x_i][y_i], sim[x_i][y_i]))

# prepare_sample_input()
# test_mask_zero()
# test_sim_cube()
# test_get_focus_mask()
test_focuscube_layer_content()

# prepare data
# preprocessor = mz.preprocessors.BasicPreprocessor(fixed_length_left=fixed_length_left,
                                                  # fixed_length_right=fixed_length_right,
                                                  # remove_stop_words=False,
                                                  # filter_low_freq=10)

# train_X, train_Y_, preprocessor = load_train_data(preprocessor, False)
# val_X, val_Y = load_val_data(preprocessor)
# pred_X1, pred_Y1 = load_test_data(preprocessor, True)
# pred_X2, pred_Y2 = load_test_data(preprocessor, False)

# train_Y = to_categorical(train_Y_)
# print(train_Y)


# import csv
# file_path = "~/datasets/pretrain_emb/glove.840B.300d.txt"
# data = pd.read_table(file_path,
                     # sep=" ",
                     # index_col=0,
                     # header=None,
                     # quoting=csv.QUOTE_NONE)

# googlenews_embedding = mz.embedding.Embedding(data)
# embedding_matrix = googlenews_embedding.build_matrix(preprocessor.context['vocab_unit'].state['term_index'], initializer=lambda: 0)

# model.load_embedding_matrix(embedding_matrix)

# history = model.fit(x = [train_X['text_left'],
                         # train_X['text_right']],                  # (20360, 1000)
                    # y = train_Y,                                  # (20360, 2)
                    # validation_data = (val_X, val_Y),
                    # batch_size = batch_size,
                    # epochs = epochs,
                    # callbacks=[MapMrrCallback(pred_X1, pred_Y1)]
                    # )

# ###############################################
# # evaluation with MAP & MRR

# for X, Y in [[pred_X1, pred_Y1], [pred_X2, pred_Y2], [train_X, train_Y_]]:
    # print("Predict dataset size: ", Y.shape)

    # num_pred_data = Y.shape[0]
    # pred_y_full = model.predict(X, batch_size=32)
    # pred_y = np.argmax(pred_y_full, axis=-1)[:, np.newaxis]     # (num_pred, 1)

    # print("Y: num_pos: {} num_neg: {}".format((Y == 1).sum(), (Y == 0).sum()))

    # qids = X["id_left"]                 # (6165, )
    # predictions = pred_y_full[:, 1]     # (6165, ) pred value for label 1
    # labels = Y[:, 0]                    # (6165, )
    # print("qids: {}, predictions: {}, labels: {}".format(qids.shape, predictions.shape, labels.shape))

    # mAP, mrr = get_map_mrr(qids, predictions, labels)

    # correct_idx = (pred_y == Y)
    # wrong_idx = (pred_y != Y)
    # print("====================================")
    # print("true pos:", ((pred_y == 1) * correct_idx).sum(), " || false pos:", ((pred_y == 1) * wrong_idx).sum())
    # print("true neg:", ((pred_y == 0) * correct_idx).sum(), " || false neg:", ((pred_y == 0) * wrong_idx).sum())
    # print("====================================")
    # print("mAP: ", mAP)
    # print("mrr: ", mrr)
    # print("====================================")
    # print()
