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
            merge_mode=None)

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
        print("inside cos: ", tensor1.shape, tensor2.shape, self._calc_dot_prod(tensor1, tensor2).shape)
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
        print("cos: ", self._calc_cos_distance(tensor1, tensor2).shape)
        return K.stack([self._calc_cos_distance(tensor1, tensor2),
                        self._calc_l2_distance(tensor1, tensor2),
                        self._calc_dot_prod(tensor1, tensor2)], axis=1)

    def _make_sim_cube_layer(self) -> keras.layers.Layer:
        """
        Pairwise word interaction modeling layer

        The layer expects input: [tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]
        with shape (B, T1, H), (B, T1, H), (B, T2, H), (B, T2, H) outputs
        tf.Tensor with shape (B, 13, T1, T2)

        :return: keras.layers.Layer
        """
        def _get_sim_cube(hs):
            print("called")
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

            print("bi_cou: ", bi_cou)
            print("for_cou: ", for_cou)
            print("back_cou: ", back_cou)
            print("add_cou: ", add_cou)
            print("masks: ", masks)
            return K.concatenate([masks, bi_cou, for_cou, back_cou, add_cou], axis=1)

        def _get_output_shape(hs_shape):
            h1for_shape, h1back_shape, h2for_shape, h2back_shape = hs_shape
            print(h1for_shape, h2for_shape)
            assert h1for_shape == h1back_shape
            assert h2for_shape == h2back_shape

            return (h1for_shape[0], 13, h1for_shape[1], h2for_shape[1]) # (13, T1, T2)

        return keras.layers.Lambda(function=_get_sim_cube, 
                                   output_shape=_get_output_shape)

    def _make_focus_cube_layer(self) -> keras.layers.Layer:
        """
        Similary focus layer

        The layer takes the output tensor produced by PWIM._make_sim_cube_layer
        as input, which has shape (B, 13, T1, T2), then generates a tensor with
        same shape yet put emphasis on highly similar word pairs
        """
        def _get_focus_mask(sim_tensor):
            """
            Generate focus mask according to the passed in similarity tensor

            :param sim_tensor: tf.Tensor with shape (B, T1, T2)
                either cos similarity or l2 similarity based on h1add and h2add
            :param mask: tf.Tensor with shape (B, T1, T2)
                the computation would based on the passed in mask. That is the 
                entries already set to be 1 in the mask would be skipped) 
            :return: tf.Tensor with shape (B, T1, T2), with focus emtries set to
                be 1 and others set to be 0.1
            """
            s1tag = K.zeros_like(sim_tensor[:, 0])   # (T1)
            s2tag = K.zeros_like(sim_tensor[0, :])   # (T2)
            t1, t2 = K.int_shape(sim_tensor)
            print("+++++++++++ t1, t2", t1, t2)

            masks = 0.1 * K.ones_like(sim_tensor)    # (T1, T2), intialize masks
            sim_tensor = K.flatten(sim_tensor)    # (T1*T2)
            _, idxs = tf.nn.top_k(sim_tensor, k=K.shape(sim_tensor)[-1], sorted=True) # (T1*T2)

            for t_idx in range(t1*t2):
                pos1 = idxs[t_idx] // t1    # index for word in sentence 1
                pos2 = idxs[t_idx] % t1     # index for word in sentence 2
                if ((s1tag[pos1] + s2tag[pos2] == 0) and 
                    (masks[pos1][pos2] == 0.1)):
                    s1tag[pos1] = 1
                    s2tag[pos2] = 1
                    masks[pos1][pos2] = 1

            return masks

        def _get_focus_cube(sim_cube):
            """
            Generate focus cube using _generate_focus_mask function above

            :param sim_cube: tf.Tensor with shape (B, 13, T1, T2)
            :return: tf.Tensor with shape (B, 13, T1, T2), sim_cube * mask
            """
            # masks = 0.1 * K.ones_like(sim_cube[0, :, :])    # (B, T1, T2), intialize masks
            print("++++++++++sim_cube:", sim_cube.shape, K.int_shape(sim_cube))

            cos_sim_tensor = sim_cube[:, 10, :, :]  # cos_similarity<h1add, h2add>
            print("++++++++++cos_sim_tensor:", cos_sim_tensor.shape, K.int_shape(cos_sim_tensor))

            masks = tf.map_fn(_get_focus_mask, cos_sim_tensor) # (B, T1, T2)

            l2_sim_tensor = sim_cube[:, 11, :, :]   # l2_similarity<h1add, h2add>
            print("++++++++++l2_sim_tensor:", l2_sim_tensor.shape, K.int_shape(l2_normalize))

            masks = tf.map_fn(_get_focus_mask, l2_sim_tensor) # (B, T1, T2)

            focus_cube = K.concatenate([sim_cube[:, :-1, :, :] * masks, 
                                        sim_cube[:, -1:, :, :]], axis=1)
            return focus_cube

        def _get_output_shape(sim_tensor_shape):
            return sim_tensor_shape

        return keras.layers.Lambda(function=_get_focus_cube, 
                                   output_shape=_get_output_shape)

    def _compute_convnet_output(self, 
                                x: tf.Tensor, 
                                filters: list) -> tf.Tensor: 
        """
        Pass the input focus_cube to the 19 layers convolution network

        :param x: tf.Tensor, shape (B, 13, T1, T2)
            (expected the output of focus_cube_layer)
        :param filters: list, number of out channels for each Conv2D layer
        :return : tf.Tensor, the final output of convnet and this PWIM
        """
        for i, f in enumerate(filters):
            x = keras.layers.Conv2D(filters=f,
                                    kernel_size=3,
                                    strides=1,
                                    padding='same',
                                    activation='relu',
                                    data_format='channels_first'
                                    )(x)  # (B, f, T1, T2)
            print(x.shape)
            x = keras.layers.MaxPooling2D(pool_size=2, 
                                          strides=2,
                                          padding='same',
                                          data_format='channels_first'
                                          )(x)
            print(x.shape)

        x = self._make_multi_layer_perceptron_layer()(x)

        # manual log_softmax output layer
        x = self._make_output_layer()(x)    # softmax activation
        x = keras.layers.Lambda(lambda t: K.log(t))(x)
        return x


    def build(self):
        """Build model."""
        # parameters
        lstm_dim = self._params['lstm_dim']

        # layers
        create_mask = keras.layers.Lambda(
            lambda x:
                K.cast(K.not_equal(x, self._params['mask_value']), K.floatx())
        )
        embedding = self._make_embedding_layer()
        lstm_layer = self._make_bilstm_layer(lstm_dim)

        # input & mask
        h1, h2 = self._make_inputs()     # [B, T_h1], [B, T_h2]
        h1_mask = create_mask(h1)         # [B, T_h1]
        h2_mask = create_mask(h2)         # [B, T_h2]
        h12_mask = keras.layers.Multiply()(        # h12_mask: [B, T_h1, T_h2]
            [self._expand_dim(h1_mask, axis=2),    # [B, T_h1, 1]
             self._expand_dim(h2_mask, axis=1)])   # [B, 1, T_h2]

        # embedding
        h1_emb = embedding(h1)   # [B, T_h1, E_dim]
        h2_emb = embedding(h2)   # [B, T_h2, E_dim]

        # context modeling
        h1_for, h1_back = lstm_layer(h1_emb)          # [B, T_h1, H*2]
        h2_for, h2_back = lstm_layer(h2_emb)          # [B, T_h2, H*2]
        
        print("after lstm: ", h1_for.shape, h1_back.shape, h2_for.shape, h2_back.shape)

        # mask a_ and b_, since the <pad> position is no more zero
        # a_ = keras.layers.Multiply()([a_, self._expand_dim(a_mask, axis=2)])
        # b_ = keras.layers.Multiply()([b_, self._expand_dim(b_mask, axis=2)])

        # pairwise word interaction modeling
        print("======== BEFORE ===========")
        sim_cube = self._make_sim_cube_layer()([h1_for, h1_back, h2_for, h2_back])
        print("========== AFTER =========")
        # sim_cube *= h12_mask
        print("============sim_cube: ", sim_cube)

        # forward pass: similarity focus layer
        focus_cube = self._make_focus_cube_layer()(sim_cube)
        # focus_cube *= h12_mask

        print("============focus_cube: ", focus_cube)
        # 19-layer conv net
        filters = [128, 164, 192, 128]
        output = self._compute_convnet_output(focus_cube, filters=filters)
        self._backend = keras.Model(inputs=[h1, h2], outputs=output)
