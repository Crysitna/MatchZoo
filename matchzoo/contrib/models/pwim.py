"""PWIM model."""
import keras
import tensorflow as tf
import keras.backend as K

import matchzoo as mz
from matchzoo.engine.param import Param
from matchzoo.engine.base_model import BaseModel
from matchzoo.engine.param_table import ParamTable


class PWIM(BaseModel):
    """
    PWIM model.
    Examples:
        >>> model = PWIM()
        >>> task = classification_task = mz.tasks.Classification(num_classes=2)
        >>> model.params['task'] = task
        >>> model.params['input_shapes'] = [(20, ), (40, )]
        >>> model.params['lstm_dim'] = 300
        >>> model.params['mlp_num_units'] = 300
        >>> model.params['embedding_input_dim'] =  5000
        >>> model.params['embedding_output_dim'] =  10
        >>> model.params['embedding_trainable'] = False
        >>> model.params['mlp_num_layers'] = 0
        >>> model.params['mlp_num_fan_out'] = 300
        >>> model.params['mlp_activation_func'] = 'tanh'
        >>> model.params['mask_value'] = 0
        >>> model.params['dropout_rate'] = 0.5
        >>> model.params['optimizer'] = keras.optimizers.Adam(lr=4e-4)
        >>> model.guess_and_fill_missing_params()
        >>> model.build()
    """

    @classmethod
    def get_default_params(cls) -> ParamTable:
        """Get default parameters."""
        params = super().get_default_params(with_embedding=True,
                                            with_multi_layer_perceptron=True)

        params.add(Param(
            name='lstm_dim',
            value=300,
            desc="The dimension of LSTM layer."
        ))

        params.add(Param(
            name='mask_value',
            value=0,
            desc="The value would be regarded as pad"
        ))

        return params

    def _expand_dim(self, inp: tf.Tensor, axis: int) -> keras.layers.Layer:
        """
        Wrap keras.backend.expand_dims into a Lambda layer.
        :param inp: input tensor to expand the dimension
        :param axis: the axis of new dimension
        """
        return keras.layers.Lambda(lambda x: K.expand_dims(x, axis=axis))(inp)

    def _make_bilstm_layer(self, lstm_dim: int) -> keras.layers.Layer:
        """
        Bidirectional LSTM layer in PWIM.
        :param lstm_dim: int, dimension of LSTM layer
        :return: `keras.layers.Layer`.
        """
        return keras.layers.Bidirectional(
            layer=keras.layers.LSTM(lstm_dim, return_sequences=True),
            merge_mode='concat')

    def _calc_l2_distance(self,
                          tensor1: tf.Tensor,
                          tensor2: tf.Tensor) -> tf.Tensor:
        """
        Calculate the L2 distance between two tensors

        :param tensor1: tf.Tensor, shape (B, T1, H)
        :param tensor2: tf.Tensor, shape (B, T2, H)
        :return: tf.Tensor, shape (B, T1, T2)
        """
        tensor1 = K.expand_dims(tensor1, axis=2) # shape (B, T1, 1, H)
        tensor2 = K.expand_dims(tensor2, axis=1) # shape (B, 1, T2, H)
        return K.sqrt(K.sum(K.square(tensor1 - tensor2), axis=-1))

    def _calc_dot_prod(self,
                       tensor1: tf.Tensor,
                       tensor2: tf.Tensor) -> tf.Tensor:
        """
        Calculate the dot product along last axis between two tensors

        :param tensor1: tf.Tensor, shape (B, T1, H)
        :param tensor2: tf.Tensor, shape (B, T2, H)
        :return: tf.Tensor, shape (B, T1, T2)
        """
        tensor2 = K.permute_dimensions(tensor2, pattern=(0, 2, 1)) # shape (B, H, T2)
        return K.batch_dot(tensor1, tensor2)

    def _calc_cos_distance(self,
                           tensor1: tf.Tensor,
                           tensor2: tf.Tensor) -> tf.Tensor:
        """
        Calculate the cosine similarity between two tensors
        
        :param tensor1: tf.Tensor, shape (B, T1, H)
        :param tensor2: tf.Tensor, shape (B, T2, H)
        :return: tf.Tensor, shape (B, T1, T2)
        """
        tensor1 = K.l2_normalize(tensor1, axis=-1)
        tensor2 = K.l2_normalize(tensor2, axis=-1)
        return self._calc_dot_prod(tensor1, tensor2)

    def _calc_cou_distance(self,
                           tensor1: tf.Tensor,
                           tensor2: tf.Tensor) -> tf.Tensor:
        """
        Stack the cos, l2, dot distance results along axis 1
        
        :param tensor1: tf.Tensor, shape (B, T1, H)
        :param tensor2: tf.Tensor, shape (B, T2, H)
        :return: tf.Tensor, shape (B, 3, T1, T2)
        """
        return K.stack([self._calc_cos_distance(tensor1, tensor2),
                        self._calc_l2_distance(tensor1, tensor2),
                        self._calc_dot_prod(tensor1, tensor2)], axis=1)

    def _make_cou_layer(self) -> keras.layers.Layer:
        """
        Comparison unit layer for modeling pairwise word interaction

        The layer expects input: [tf.Tensor, tf.Tensor] with shape
        (B, T1, H), (B, T2, H) outputs tf.Tensor with shape (B, 13, T1, T2)

        :return: keras.layers.Layer
        """
        def _get_sim_cube(hs):
            # prepare hidden layers
            h1for, h1back, h2for, h2back = hs        # (B, T1, H), (B, T2, H)
            h1bi = K.concatenate([h1for, h1back])    # (B, T1, H * 2)
            h2bi = K.concatenate([h2for, h2back])    # (B, T2, H * 2)
            h1add = h1for + h1back  # (B, T1, H)
            h2add = h2for + h2back  # (B, T2, H)

            # calculate similarities, all *_cou has shape: (B, 3, T1, T2)
            bi_cou = self._calc_cou_distance(h1bi, h2bi)
            for_cou = self._calc_cou_distance(h1for, h2for)
            back_cou = self._calc_cou_distance(h1back, h2back)
            add_cou = self._calc_cou_distance(h1add, h2add)
            masks = K.ones_like(bi_cou[:, :1, :, :])    # (B, 1, T1, T2)
            return K.concatenate([masks, bi_cou, for_cou, back_cou, add_cou], axis=1)

        return keras.layers.Lambda(_get_sim_cube)

    def build(self):
        """Build model."""
        # parameters
        lstm_dim = self._params['lstm_dim']
        dropout_rate = self._params['dropout_rate']

        # layers
        create_mask = keras.layers.Lambda(
            lambda x:
                K.cast(K.not_equal(x, self._params['mask_value']), K.floatx())
        )
        embedding = self._make_embedding_layer()
        lstm_compare = self._make_bilstm_layer(lstm_dim)
        lstm_compose = self._make_bilstm_layer(lstm_dim)
        dense_compare = keras.layers.Dense(units=lstm_dim,
                                           activation='relu',
                                           use_bias=True)
        dropout = keras.layers.Dropout(dropout_rate)

        # model
        a, b = self._make_inputs()      # [B, T_a], [B, T_b]
        a_mask = create_mask(a)         # [B, T_a]
        b_mask = create_mask(b)         # [B, T_b]

        # encoding
        a_emb = dropout(embedding(a))   # [B, T_a, E_dim]
        b_emb = dropout(embedding(b))   # [B, T_b, E_dim]

        a_ = lstm_compare(a_emb)          # [B, T_a, H*2]
        b_ = lstm_compare(b_emb)          # [B, T_b, H*2]

        # mask a_ and b_, since the <pad> position is no more zero
        a_ = keras.layers.Multiply()([a_, self._expand_dim(a_mask, axis=2)])
        b_ = keras.layers.Multiply()([b_, self._expand_dim(b_mask, axis=2)])

        # local inference
        e = keras.layers.Dot(axes=-1)([a_, b_])   # [B, T_a, T_b]
        _ab_mask = keras.layers.Multiply()(       # _ab_mask: [B, T_a, T_b]
            [self._expand_dim(a_mask, axis=2),    # [B, T_a, 1]
             self._expand_dim(b_mask, axis=1)])   # [B, 1, T_b]

        pm = keras.layers.Permute((2, 1))
        mask_layer = self._make_atten_mask_layer()
        softmax_layer = keras.layers.Softmax(axis=-1)

        e_a = softmax_layer(mask_layer([e, _ab_mask]))          # [B, T_a, T_b]
        e_b = softmax_layer(mask_layer([pm(e), pm(_ab_mask)]))  # [B, T_b, T_a]

        # alignment (a_t = a~, b_t = b~ )
        a_t = keras.layers.Dot(axes=(2, 1))([e_a, b_])   # [B, T_a, H*2]
        b_t = keras.layers.Dot(axes=(2, 1))([e_b, a_])   # [B, T_b, H*2]

        # local inference info enhancement
        m_a = keras.layers.Concatenate(axis=-1)([
            a_,
            a_t,
            keras.layers.Subtract()([a_, a_t]),
            keras.layers.Multiply()([a_, a_t])])    # [B, T_a, H*2*4]
        m_b = keras.layers.Concatenate(axis=-1)([
            b_,
            b_t,
            keras.layers.Subtract()([b_, b_t]),
            keras.layers.Multiply()([b_, b_t])])    # [B, T_b, H*2*4]

        # project m_a and m_b from 4*H*2 dim to H dim
        m_a = dropout(dense_compare(m_a))   # [B, T_a, H]
        m_b = dropout(dense_compare(m_b))   # [B, T_a, H]

        # inference composition
        v_a = lstm_compose(m_a)          # [B, T_a, H*2]
        v_b = lstm_compose(m_b)          # [B, T_b, H*2]

        # pooling
        v_a = keras.layers.Concatenate(axis=-1)(
            [self._avg(v_a, a_mask), self._max(v_a, a_mask)])   # [B, H*4]
        v_b = keras.layers.Concatenate(axis=-1)(
            [self._avg(v_b, b_mask), self._max(v_b, b_mask)])   # [B, H*4]
        v = keras.layers.Concatenate(axis=-1)([v_a, v_b])       # [B, H*8]

        # mlp (multilayer perceptron) classifier
        output = self._make_multi_layer_perceptron_layer()(v)  # [B, H]
        output = dropout(output)
        output = self._make_output_layer()(output)             # [B, #classes]

        self._backend = keras.Model(inputs=[a, b], outputs=output)
