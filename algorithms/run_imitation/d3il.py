import numpy as np
import tensorflow as tf
from buffers import VisualReplayBuffer
from collections import deque


class Encoder(tf.keras.layers.Layer):
    """Preprocessor outputting latent space samples."""
    def __init__(self, layers):
        super(Encoder, self).__init__()
        self._layers = layers

    def layers_out(self, inputs):
        out = inputs
        for layer in self._layers:
            out = layer(out)
        return out

    @tf.function
    def call(self, inputs):
        out = self.layers_out(inputs)
        return out


class Generator(tf.keras.layers.Layer):
    """
    A normal generator
    Generate images using domain feature & expert feature
    """
    def __init__(self, layers, past_frames, n_input_channels):
        super(Generator, self).__init__()
        self._layers = layers
        self._past_frames = past_frames
        self._n_input_channels = n_input_channels

    def layers_out(self, inputs_d, inputs_e):
        out_d = inputs_d
        out = inputs_e
        for layer in self._layers:
            im_size = out.shape[1]
            out_d = tf.reshape(tf.tile(inputs_d, (1, im_size ** 2)), [out.shape[0], im_size, im_size, out_d.shape[-1]])
            out = tf.concat([out_d, out], axis=-1)
            out = layer(out)
        return out

    def get_im_size(self, shape):
        return int(pow(float(shape[1] // self._n_input_channels), 0.5))

    @tf.function
    def call(self, inputs_d, inputs_e):
        input_shape = inputs_e.get_shape()
        im_size = self.get_im_size(input_shape)
        feat_d = tf.reshape(inputs_d, [input_shape[0], -1])
        feat_e = tf.reshape(inputs_e, [input_shape[0], im_size, im_size, self._n_input_channels])
        out = self.layers_out(feat_d, feat_e)
        return out


class VisualDiscriminator(tf.keras.layers.Layer):
    """Discriminator with support for visual observations."""
    def __init__(self, layers, stab_const=0.0, rew='mixed'):
        super(VisualDiscriminator, self).__init__()
        self._dis_layers = layers
        self._sb = stab_const
        self._rew = rew

    @tf.function
    def call(self, inputs):
        out = inputs
        for layer in self._dis_layers:
            out = layer(out)
        return out

    def get_prob(self, ims):
        model_out = self.__call__(ims)
        return tf.reshape(tf.sigmoid(model_out), [-1])

    def get_reward(self, ims):
        if self._rew == 'positive':
            return -1 * tf.math.log(1 - self.get_prob(ims) + self._sb)
        elif self._rew == 'negative':
            return tf.math.log(self.get_prob(ims) + self._sb)
        return (tf.math.log(self.get_prob(ims) + self._sb) -
                tf.math.log(1 - self.get_prob(ims) + self._sb))


class InvariantDiscriminator(VisualDiscriminator):
    """Invariant discriminator model."""
    def __init__(self, layers, stab_const=0.0, rew='mixed'):
        super(InvariantDiscriminator, self).__init__(layers, stab_const, rew)


class ExpertFeatureDiscriminator(VisualDiscriminator):
    """Invariant discriminator model."""
    def __init__(self, layers, stab_const=0.0, rew='mixed'):
        super(ExpertFeatureDiscriminator, self).__init__(layers, stab_const, rew)


class TranslatedImageDiscriminator(tf.keras.layers.Layer):
    """Discriminator with support for visual observations."""
    def __init__(self, layers, stab_const=0.0, rew='mixed'):
        super(TranslatedImageDiscriminator, self).__init__()
        self._dis_layers = layers
        self._sb = stab_const
        self._rew = rew

    @tf.function
    def call(self, inputs):
        out = inputs
        for layer in self._dis_layers:
            out = layer(out)
        return out


class ExpertImageDiscriminator(tf.keras.layers.Layer):
    """Discriminator with support for visual observations."""
    def __init__(self, layers, stab_const=0.0, rew='mixed'):
        super(ExpertImageDiscriminator, self).__init__()
        self._dis_layers = layers
        self._sb = stab_const
        self._rew = rew

    @tf.function
    def call(self, inputs):
        out = inputs
        for layer in self._dis_layers:
            out = layer(out)
        return out


class CustomReplayBuffer(VisualReplayBuffer):
    """Replay buffer computing calculating the pseudo-rewards from a discriminator."""
    def __init__(self, model, buffer_size, initial_data={}, done_reward=None, rew_multiplier=None):
        super(CustomReplayBuffer, self).__init__(buffer_size, initial_data)
        self.model = model
        self._done_reward = done_reward
        self._rew_multiplier = rew_multiplier

    def get_random_batch(self, batch_size, re_eval_rw=True):
        """Get random batch of data.

        Parameters
        ----------
        batch_size : Batch size of experience to collect.
        re_eval_rw : Compute pseudo-rewards for batch, default is True.
        """
        out = super(CustomReplayBuffer, self).get_random_batch(batch_size)
        if re_eval_rw:
            if self._done_reward is not None:   # UMaze environments
                out['rew'] = self._rew_multiplier * (self.model.get_reward(out['ims']) + self._done_reward * np.reshape(out['don'].astype(np.float32), (batch_size, -1)))
            else:
                out['rew'] = self.model.get_reward(out['ims'])
        return out


# TODO: Type 1
# ==================================================
# ==================================================
class D3ILModel(tf.keras.Model):
    """
    Imitation - Type 1
    Encoder_d * 2 (source, target)
    Encoder_e * 2 (source, target)
    Generator * 2 (source, target)
    Feature discriminator * 2 (domain, expert)
    Translation discriminator * 2 (source, target)
    Expert discriminator * 1 (expert vs. non-expert)
    """
    def __init__(self,
                 agent,
                 make_encoder_d_fn,
                 make_encoder_e_fn,
                 make_generator_fn,
                 make_dom_disc_fn,
                 make_cls_disc_fn,
                 make_trans_disc_fn,
                 make_expert_disc_fn,
                 c_gan_trans=1.0,
                 c_gan_feat=1.0,
                 c_recon=1.0,
                 c_cycle=1.0,
                 c_feat_mean=1.0,
                 c_feat_recon=1.0,
                 c_feat_reg=0.0,
                 c_feat_cycle=1.0,
                 c_norm_de=1,
                 c_norm_be=1,
                 type_recon_loss='l2',
                 eg_update_interval=1,
                 it_max_grad_norm=None,
                 it_lr=0.0003,
                 d_rew='mixed',
                 d_max_grad_norm=None,
                 d_learning_rate=0.0003,
                 past_frames=4,
                 optimizer=None):
        super(D3ILModel, self).__init__()
        # Source domain
        self._source_encoder_d = make_encoder_d_fn()
        self._source_encoder_e = make_encoder_e_fn()
        # Target domain
        self._target_encoder_d = make_encoder_d_fn()
        self._target_encoder_e = make_encoder_e_fn()
        # Generator
        self._source_generator = make_generator_fn()
        self._target_generator = make_generator_fn()
        # Discrminator (feat)
        self._feat_e_dom_discriminator = make_dom_disc_fn()
        self._feat_d_cls_discriminator = make_cls_disc_fn()
        self._feat_e_cls_discriminator = make_cls_disc_fn()
        self._feat_d_dom_discriminator = make_dom_disc_fn()
        # Discriminator (trans)
        self._source_trans_discriminator = make_trans_disc_fn()
        self._target_trans_discriminator = make_trans_disc_fn()
        # Discrminator (real)
        self._expert_discriminator = None
        self._epsilon = 1e-7
        # Agent
        self._agent = agent

        # Common
        self._past_frames = past_frames
        self._it_lr = it_lr
        self._it_optimizer = tf.keras.optimizers.Adam(it_lr)
        self._d_optimizer = tf.keras.optimizers.Adam(d_learning_rate)
        self._gp_lambda = 10.0
        self._gp_lambda_reward = 100.0
        self._rew = d_rew

        # Debugging: Coefficients
        self.c_gan_trans = c_gan_trans
        self.c_gan_feat = c_gan_feat
        self.c_recon = c_recon
        self.c_cycle = c_cycle
        self.c_feat_mean = c_feat_mean
        self.c_feat_recon = c_feat_recon
        self.c_feat_reg = c_feat_reg
        self.c_feat_cycle = c_feat_cycle

        self.c_norm_de = c_norm_de
        self.c_norm_be = c_norm_be

        self.type_recon_loss = type_recon_loss
        self.eg_update_interval = eg_update_interval
        self.it_max_grad_norm = it_max_grad_norm
        self.d_max_grad_norm = d_max_grad_norm
        self.d_learning_rate = d_learning_rate

        # Debugging: Update rule
        self.expert_d_update_rule = 'gan'
        self.it_model_gan_update_rule = 'wgan'

    # Phase 1
    # ==================================================
    @tf.function
    def image_translation(self, se_inputs, sn_inputs, tn_inputs, tl_inputs, se_masks, sn_masks, tn_masks, tl_masks):
        """
        Perform image translation, and return the outputs of encoders, generators and discriminators.
        """
        # domain feature extraction
        se_enc_d_out = self._source_encoder_d(se_inputs[:, :, :, 0:3])
        sn_enc_d_out = self._source_encoder_d(sn_inputs[:, :, :, 0:3])
        tn_enc_d_out = self._target_encoder_d(tn_inputs[:, :, :, 0:3])
        tl_enc_d_out = self._target_encoder_d(tl_inputs[:, :, :, 0:3])

        # behavior feature extraction
        # ("e" refers to "expertness feature", which was a former name of "behavior feature")
        se_enc_e_out = self._source_encoder_e(se_inputs)
        sn_enc_e_out = self._source_encoder_e(sn_inputs)
        tn_enc_e_out = self._target_encoder_e(tn_inputs)
        tl_enc_e_out = self._target_encoder_e(tl_inputs)

        # ==================================================
        # domain independence discrimination
        se_e_dom_disc_logits = self._feat_e_dom_discriminator(se_enc_e_out)
        sn_e_dom_disc_logits = self._feat_e_dom_discriminator(sn_enc_e_out)
        tn_e_dom_disc_logits = self._feat_e_dom_discriminator(tn_enc_e_out)
        tl_e_dom_disc_logits = self._feat_e_dom_discriminator(tl_enc_e_out)

        # class (behavior) independence discrimination
        se_d_cls_disc_logits = self._feat_d_cls_discriminator(se_enc_d_out)
        sn_d_cls_disc_logits = self._feat_d_cls_discriminator(sn_enc_d_out)
        tn_d_cls_disc_logits = self._feat_d_cls_discriminator(tn_enc_d_out)
        tl_d_cls_disc_logits = self._feat_d_cls_discriminator(tl_enc_d_out)

        # class (behavior) prediction discrimination
        se_e_cls_disc_logits = self._feat_e_cls_discriminator(se_enc_e_out)
        sn_e_cls_disc_logits = self._feat_e_cls_discriminator(sn_enc_e_out)
        tn_e_cls_disc_logits = self._feat_e_cls_discriminator(tn_enc_e_out)
        tl_e_cls_disc_logits = self._feat_e_cls_discriminator(tl_enc_e_out)

        # domain prediction discrimination
        se_d_dom_disc_logits = self._feat_d_dom_discriminator(se_enc_d_out)
        sn_d_dom_disc_logits = self._feat_d_dom_discriminator(sn_enc_d_out)
        tn_d_dom_disc_logits = self._feat_d_dom_discriminator(tn_enc_d_out)
        tl_d_dom_disc_logits = self._feat_d_dom_discriminator(tl_enc_d_out)

        # ==================================================
        # image-to-image translation
        # 1. (se <=> sn)
        se0sn_trans_out = self._source_generator(se_enc_d_out, sn_enc_e_out) * sn_masks + (1.0 - sn_masks) * 0.5 / 256  # se(s) + sn(n)
        sn0se_trans_out = self._source_generator(sn_enc_d_out, se_enc_e_out) * se_masks + (1.0 - se_masks) * 0.5 / 256  # sn(s) + se(e)

        # 2. (se <=> tn)
        se0tn_trans_out = self._source_generator(se_enc_d_out, tn_enc_e_out) * tn_masks + (1.0 - tn_masks) * 0.5 / 256  # se(s) + tn(n)
        tn0se_trans_out = self._target_generator(tn_enc_d_out, se_enc_e_out) * se_masks + (1.0 - se_masks) * 0.5 / 256  # tn(t) + se(e)

        # 3. (se <=> tl)
        se0tl_trans_out = self._source_generator(se_enc_d_out, tl_enc_e_out) * tl_masks + (1.0 - tl_masks) * 0.5 / 256  # se(s) + tl(l)
        tl0se_trans_out = self._target_generator(tl_enc_d_out, se_enc_e_out) * se_masks + (1.0 - se_masks) * 0.5 / 256  # tl(t) + se(e)

        # 4. (sn <=> tn)
        sn0tn_trans_out = self._source_generator(sn_enc_d_out, tn_enc_e_out) * tn_masks + (1.0 - tn_masks) * 0.5 / 256  # sn(s) + tn(n)
        tn0sn_trans_out = self._target_generator(tn_enc_d_out, sn_enc_e_out) * sn_masks + (1.0 - sn_masks) * 0.5 / 256  # tn(t) + sn(n)

        # 5. (sn <=> tl)
        sn0tl_trans_out = self._source_generator(sn_enc_d_out, tl_enc_e_out) * tl_masks + (1.0 - tl_masks) * 0.5 / 256  # sn(s) + tl(l)
        tl0sn_trans_out = self._target_generator(tl_enc_d_out, sn_enc_e_out) * sn_masks + (1.0 - sn_masks) * 0.5 / 256  # tl(t) + sn(n)

        # 6. (tn <=> tl)
        tn0tl_trans_out = self._target_generator(tn_enc_d_out, tl_enc_e_out) * tl_masks + (1.0 - tl_masks) * 0.5 / 256  # tn(t) + tl(l)
        tl0tn_trans_out = self._target_generator(tl_enc_d_out, tn_enc_e_out) * tn_masks + (1.0 - tn_masks) * 0.5 / 256  # tl(t) + tn(n)


        # ==================================================
        # original/translated image discrimination
        se_trans_disc_logits = self._source_trans_discriminator(se_inputs)
        sn_trans_disc_logits = self._source_trans_discriminator(sn_inputs)
        tn_trans_disc_logits = self._target_trans_discriminator(tn_inputs)
        tl_trans_disc_logits = self._target_trans_discriminator(tl_inputs)
        se0sn_trans_disc_logits = self._source_trans_discriminator(se0sn_trans_out)  # 1
        sn0se_trans_disc_logits = self._source_trans_discriminator(sn0se_trans_out)  # 1
        se0tn_trans_disc_logits = self._source_trans_discriminator(se0tn_trans_out)  # 2
        tn0se_trans_disc_logits = self._target_trans_discriminator(tn0se_trans_out)  # 2
        se0tl_trans_disc_logits = self._source_trans_discriminator(se0tl_trans_out)  # 3
        tl0se_trans_disc_logits = self._target_trans_discriminator(tl0se_trans_out)  # 3
        sn0tn_trans_disc_logits = self._source_trans_discriminator(sn0tn_trans_out)  # 4
        tn0sn_trans_disc_logits = self._target_trans_discriminator(tn0sn_trans_out)  # 4
        sn0tl_trans_disc_logits = self._source_trans_discriminator(sn0tl_trans_out)  # 5
        tl0sn_trans_disc_logits = self._target_trans_discriminator(tl0sn_trans_out)  # 5
        tn0tl_trans_disc_logits = self._target_trans_discriminator(tn0tl_trans_out)  # 6
        tl0tn_trans_disc_logits = self._target_trans_discriminator(tl0tn_trans_out)  # 6

        # ==================================================
        # cycle consistency
        # 1. (se <=> sn)
        se0sn_trans_enc_d_out = self._source_encoder_d(se0sn_trans_out[:, :, :, 0:3])  # s
        se0sn_trans_enc_e_out = self._source_encoder_e(se0sn_trans_out)  # n
        sn0se_trans_enc_d_out = self._source_encoder_d(sn0se_trans_out[:, :, :, 0:3])  # s
        sn0se_trans_enc_e_out = self._source_encoder_e(sn0se_trans_out)  # e
        se0sn_sn0se_cycle_out = self._source_generator(se0sn_trans_enc_d_out, sn0se_trans_enc_e_out) * se_masks + (1.0 - se_masks) * 0.5 / 256  # (sn)(s)+(se)(e)
        sn0se_se0sn_cycle_out = self._source_generator(sn0se_trans_enc_d_out, se0sn_trans_enc_e_out) * sn_masks + (1.0 - sn_masks) * 0.5 / 256  # (se)(s)+(sn)(n)

        # 2. (se <=> tn)
        se0tn_trans_enc_d_out = self._source_encoder_d(se0tn_trans_out[:, :, :, 0:3])  # s
        se0tn_trans_enc_e_out = self._source_encoder_e(se0tn_trans_out)  # n
        tn0se_trans_enc_d_out = self._target_encoder_d(tn0se_trans_out[:, :, :, 0:3])  # t
        tn0se_trans_enc_e_out = self._target_encoder_e(tn0se_trans_out)  # e
        se0tn_tn0se_cycle_out = self._source_generator(se0tn_trans_enc_d_out, tn0se_trans_enc_e_out) * se_masks + (1.0 - se_masks) * 0.5 / 256  # (sn)(s)+(te)(e)
        tn0se_se0tn_cycle_out = self._target_generator(tn0se_trans_enc_d_out, se0tn_trans_enc_e_out) * tn_masks + (1.0 - tn_masks) * 0.5 / 256  # (te)(t)+(sn)(n)

        # 3. (se <=> tl)
        se0tl_trans_enc_d_out = self._source_encoder_d(se0tl_trans_out[:, :, :, 0:3])  # s
        se0tl_trans_enc_e_out = self._source_encoder_e(se0tl_trans_out)  # l
        tl0se_trans_enc_d_out = self._target_encoder_d(tl0se_trans_out[:, :, :, 0:3])  # t
        tl0se_trans_enc_e_out = self._target_encoder_e(tl0se_trans_out)  # e
        se0tl_tl0se_cycle_out = self._source_generator(se0tl_trans_enc_d_out, tl0se_trans_enc_e_out) * se_masks + (1.0 - se_masks) * 0.5 / 256  # (sl)(s)+(te)(e)
        tl0se_se0tl_cycle_out = self._target_generator(tl0se_trans_enc_d_out, se0tl_trans_enc_e_out) * tl_masks + (1.0 - tl_masks) * 0.5 / 256  # (te)(t)+(sl)(l)

        # 4. (sn <=> tn)
        sn0tn_trans_enc_d_out = self._source_encoder_d(sn0tn_trans_out[:, :, :, 0:3])  # s
        sn0tn_trans_enc_e_out = self._source_encoder_e(sn0tn_trans_out)  # n
        tn0sn_trans_enc_d_out = self._target_encoder_d(tn0sn_trans_out[:, :, :, 0:3])  # t
        tn0sn_trans_enc_e_out = self._target_encoder_e(tn0sn_trans_out)  # n
        sn0tn_tn0sn_cycle_out = self._source_generator(sn0tn_trans_enc_d_out, tn0sn_trans_enc_e_out) * sn_masks + (1.0 - sn_masks) * 0.5 / 256  # (sn)(s)+(tn)(n)
        tn0sn_sn0tn_cycle_out = self._target_generator(tn0sn_trans_enc_d_out, sn0tn_trans_enc_e_out) * tn_masks + (1.0 - tn_masks) * 0.5 / 256  # (tn)(t)+(sn)(n)

        # 5. (sn <=> tl)
        sn0tl_trans_enc_d_out = self._source_encoder_d(sn0tl_trans_out[:, :, :, 0:3])  # s
        sn0tl_trans_enc_e_out = self._source_encoder_e(sn0tl_trans_out)  # l
        tl0sn_trans_enc_d_out = self._target_encoder_d(tl0sn_trans_out[:, :, :, 0:3])  # t
        tl0sn_trans_enc_e_out = self._target_encoder_e(tl0sn_trans_out)  # n
        sn0tl_tl0sn_cycle_out = self._source_generator(sn0tl_trans_enc_d_out, tl0sn_trans_enc_e_out) * sn_masks + (1.0 - sn_masks) * 0.5 / 256  # (sl)(s)+(tn)(n)
        tl0sn_sn0tl_cycle_out = self._target_generator(tl0sn_trans_enc_d_out, sn0tl_trans_enc_e_out) * tl_masks + (1.0 - tl_masks) * 0.5 / 256  # (tn)(t)+(sl)(l)

        # 6. (tn <=> tl)
        tn0tl_trans_enc_d_out = self._target_encoder_d(tn0tl_trans_out[:, :, :, 0:3])  # t
        tn0tl_trans_enc_e_out = self._target_encoder_e(tn0tl_trans_out)  # l
        tl0tn_trans_enc_d_out = self._target_encoder_d(tl0tn_trans_out[:, :, :, 0:3])  # t
        tl0tn_trans_enc_e_out = self._target_encoder_e(tl0tn_trans_out)  # n
        tn0tl_tl0tn_cycle_out = self._target_generator(tn0tl_trans_enc_d_out, tl0tn_trans_enc_e_out) * tn_masks + (1.0 - tn_masks) * 0.5 / 256  # (tl)(t)+(tn)(n)
        tl0tn_tn0tl_cycle_out = self._target_generator(tl0tn_trans_enc_d_out, tn0tl_trans_enc_e_out) * tl_masks + (1.0 - tl_masks) * 0.5 / 256  # (tn)(t)+(tl)(l)

        # ==================================================
        # Reconstruction
        se0se_recon_out = self._source_generator(se_enc_d_out, se_enc_e_out) * se_masks + (1.0 - se_masks) * 0.5 / 256
        sn0sn_recon_out = self._source_generator(sn_enc_d_out, sn_enc_e_out) * sn_masks + (1.0 - sn_masks) * 0.5 / 256
        tn0tn_recon_out = self._target_generator(tn_enc_d_out, tn_enc_e_out) * tn_masks + (1.0 - tn_masks) * 0.5 / 256
        tl0tl_recon_out = self._target_generator(tl_enc_d_out, tl_enc_e_out) * tl_masks + (1.0 - tl_masks) * 0.5 / 256

        se0se_recon_enc_d_out = self._source_encoder_d(se0se_recon_out[:, :, :, 0:3])
        sn0sn_recon_enc_d_out = self._source_encoder_d(sn0sn_recon_out[:, :, :, 0:3])
        tn0tn_recon_enc_d_out = self._target_encoder_d(tn0tn_recon_out[:, :, :, 0:3])
        tl0tl_recon_enc_d_out = self._target_encoder_d(tl0tl_recon_out[:, :, :, 0:3])

        se0se_recon_enc_e_out = self._source_encoder_e(se0se_recon_out)
        sn0sn_recon_enc_e_out = self._source_encoder_e(sn0sn_recon_out)
        tn0tn_recon_enc_e_out = self._target_encoder_e(tn0tn_recon_out)
        tl0tl_recon_enc_e_out = self._target_encoder_e(tl0tl_recon_out)

        se0se_recon_disc_logits = self._source_trans_discriminator(se_inputs)
        sn0sn_recon_disc_logits = self._source_trans_discriminator(sn_inputs)
        tn0tn_recon_disc_logits = self._target_trans_discriminator(tn_inputs)
        tl0tl_recon_disc_logits = self._target_trans_discriminator(tl_inputs)

        return se_enc_d_out, sn_enc_d_out, tn_enc_d_out, tl_enc_d_out, \
               se_enc_e_out, sn_enc_e_out, tn_enc_e_out, tl_enc_e_out, \
               se_e_dom_disc_logits, sn_e_dom_disc_logits, tn_e_dom_disc_logits, tl_e_dom_disc_logits, \
               se_d_cls_disc_logits, sn_d_cls_disc_logits, tn_d_cls_disc_logits, tl_d_cls_disc_logits, \
               se_trans_disc_logits, sn_trans_disc_logits, tn_trans_disc_logits, tl_trans_disc_logits, \
               se0sn_trans_out, sn0se_trans_out, se0tn_trans_out, tn0se_trans_out, \
               se0tl_trans_out, tl0se_trans_out, sn0tn_trans_out, tn0sn_trans_out, \
               sn0tl_trans_out, tl0sn_trans_out, tn0tl_trans_out, tl0tn_trans_out, \
               se0sn_trans_disc_logits, sn0se_trans_disc_logits, se0tn_trans_disc_logits, tn0se_trans_disc_logits, \
               se0tl_trans_disc_logits, tl0se_trans_disc_logits, sn0tn_trans_disc_logits, tn0sn_trans_disc_logits, \
               sn0tl_trans_disc_logits, tl0sn_trans_disc_logits, tn0tl_trans_disc_logits, tl0tn_trans_disc_logits, \
               se0sn_sn0se_cycle_out, sn0se_se0sn_cycle_out, se0tn_tn0se_cycle_out, tn0se_se0tn_cycle_out, \
               se0tl_tl0se_cycle_out, tl0se_se0tl_cycle_out, sn0tn_tn0sn_cycle_out, tn0sn_sn0tn_cycle_out, \
               sn0tl_tl0sn_cycle_out, tl0sn_sn0tl_cycle_out, tn0tl_tl0tn_cycle_out, tl0tn_tn0tl_cycle_out, \
               se0se_recon_out, sn0sn_recon_out, tn0tn_recon_out, tl0tl_recon_out, \
               se0se_recon_enc_d_out, sn0sn_recon_enc_d_out, tn0tn_recon_enc_d_out, tl0tl_recon_enc_d_out, \
               se0se_recon_enc_e_out, sn0sn_recon_enc_e_out, tn0tn_recon_enc_e_out, tl0tl_recon_enc_e_out, \
               se0se_recon_disc_logits, sn0sn_recon_disc_logits, tn0tn_recon_disc_logits, tl0tl_recon_disc_logits, \
               se_e_cls_disc_logits, sn_e_cls_disc_logits, tn_e_cls_disc_logits, tl_e_cls_disc_logits, \
               se_d_dom_disc_logits, sn_d_dom_disc_logits, tn_d_dom_disc_logits, tl_d_dom_disc_logits, \
               se0sn_trans_enc_d_out, se0sn_trans_enc_e_out, sn0se_trans_enc_d_out, sn0se_trans_enc_e_out, \
               se0tn_trans_enc_d_out, se0tn_trans_enc_e_out, tn0se_trans_enc_d_out, tn0se_trans_enc_e_out, \
               se0tl_trans_enc_d_out, se0tl_trans_enc_e_out, tl0se_trans_enc_d_out, tl0se_trans_enc_e_out, \
               sn0tn_trans_enc_d_out, sn0tn_trans_enc_e_out, tn0sn_trans_enc_d_out, tn0sn_trans_enc_e_out, \
               sn0tl_trans_enc_d_out, sn0tl_trans_enc_e_out, tl0sn_trans_enc_d_out, tl0sn_trans_enc_e_out, \
               tn0tl_trans_enc_d_out, tn0tl_trans_enc_e_out, tl0tn_trans_enc_d_out, tl0tn_trans_enc_e_out

    def train_image_translation(self, _se_ims, _sn_ims, _tn_ims, _tl_ims, epoch):
        """
        Train the image translation model (for pre-training and/or tuning).
        """
        # Reshape image
        se_ims = self.reshape_input_images(_se_ims)
        sn_ims = self.reshape_input_images(_sn_ims)
        tn_ims = self.reshape_input_images(_tn_ims)
        tl_ims = self.reshape_input_images(_tl_ims)

        # Get mask (zero value for images before t=0)
        se_masks = self.get_masks(se_ims)
        sn_masks = self.get_masks(sn_ims)
        tn_masks = self.get_masks(tn_ims)
        tl_masks = self.get_masks(tl_ims)

        with tf.GradientTape(persistent=True) as tape:
            # forward
            se_enc_d_out, sn_enc_d_out, tn_enc_d_out, tl_enc_d_out, \
            se_enc_e_out, sn_enc_e_out, tn_enc_e_out, tl_enc_e_out, \
            se_e_dom_disc_logits, sn_e_dom_disc_logits, tn_e_dom_disc_logits, tl_e_dom_disc_logits, \
            se_d_cls_disc_logits, sn_d_cls_disc_logits, tn_d_cls_disc_logits, tl_d_cls_disc_logits, \
            se_trans_disc_logits, sn_trans_disc_logits, tn_trans_disc_logits, tl_trans_disc_logits, \
            se0sn_trans_out, sn0se_trans_out, se0tn_trans_out, tn0se_trans_out, \
            se0tl_trans_out, tl0se_trans_out, sn0tn_trans_out, tn0sn_trans_out, \
            sn0tl_trans_out, tl0sn_trans_out, tn0tl_trans_out, tl0tn_trans_out, \
            se0sn_trans_disc_logits, sn0se_trans_disc_logits, se0tn_trans_disc_logits, tn0se_trans_disc_logits, \
            se0tl_trans_disc_logits, tl0se_trans_disc_logits, sn0tn_trans_disc_logits, tn0sn_trans_disc_logits, \
            sn0tl_trans_disc_logits, tl0sn_trans_disc_logits, tn0tl_trans_disc_logits, tl0tn_trans_disc_logits, \
            se0sn_sn0se_cycle_out, sn0se_se0sn_cycle_out, se0tn_tn0se_cycle_out, tn0se_se0tn_cycle_out, \
            se0tl_tl0se_cycle_out, tl0se_se0tl_cycle_out, sn0tn_tn0sn_cycle_out, tn0sn_sn0tn_cycle_out, \
            sn0tl_tl0sn_cycle_out, tl0sn_sn0tl_cycle_out, tn0tl_tl0tn_cycle_out, tl0tn_tn0tl_cycle_out, \
            se0se_recon_out, sn0sn_recon_out, tn0tn_recon_out, tl0tl_recon_out, \
            se0se_recon_enc_d_out, sn0sn_recon_enc_d_out, tn0tn_recon_enc_d_out, tl0tl_recon_enc_d_out, \
            se0se_recon_enc_e_out, sn0sn_recon_enc_e_out, tn0tn_recon_enc_e_out, tl0tl_recon_enc_e_out, \
            se0se_recon_disc_logits, sn0sn_recon_disc_logits, tn0tn_recon_disc_logits, tl0tl_recon_disc_logits, \
            se_e_cls_disc_logits, sn_e_cls_disc_logits, tn_e_cls_disc_logits, tl_e_cls_disc_logits, \
            se_d_dom_disc_logits, sn_d_dom_disc_logits, tn_d_dom_disc_logits, tl_d_dom_disc_logits, \
            se0sn_trans_enc_d_out, se0sn_trans_enc_e_out, sn0se_trans_enc_d_out, sn0se_trans_enc_e_out, \
            se0tn_trans_enc_d_out, se0tn_trans_enc_e_out, tn0se_trans_enc_d_out, tn0se_trans_enc_e_out, \
            se0tl_trans_enc_d_out, se0tl_trans_enc_e_out, tl0se_trans_enc_d_out, tl0se_trans_enc_e_out, \
            sn0tn_trans_enc_d_out, sn0tn_trans_enc_e_out, tn0sn_trans_enc_d_out, tn0sn_trans_enc_e_out, \
            sn0tl_trans_enc_d_out, sn0tl_trans_enc_e_out, tl0sn_trans_enc_d_out, tl0sn_trans_enc_e_out, \
            tn0tl_trans_enc_d_out, tn0tl_trans_enc_e_out, tl0tn_trans_enc_d_out, tl0tn_trans_enc_e_out \
            = self.image_translation(se_ims, sn_ims, tn_ims, tl_ims, se_masks, sn_masks, tn_masks, tl_masks)

            # 1-1) behavior feature - domain adversarial loss
            if self.it_model_gan_update_rule == 'wgan':
                feat_e_dom_source_score = tf.reduce_mean(tf.concat([se_e_dom_disc_logits,
                                                                    sn_e_dom_disc_logits,], axis=0))
                feat_e_dom_target_score = tf.reduce_mean(tf.concat([tl_e_dom_disc_logits,
                                                                    tn_e_dom_disc_logits,], axis=0))
                feat_e_dom_gp_loss = self.gradient_penalty(
                    tf.concat([se_enc_e_out, sn_enc_e_out, ], axis=0),
                    tf.concat([tl_enc_e_out, tn_enc_e_out, ], axis=0),
                    self._feat_e_dom_discriminator)
                # loss for discriminators
                feat_e_dom_gan_loss_dd = - feat_e_dom_source_score + feat_e_dom_target_score + self._gp_lambda * feat_e_dom_gp_loss
                # loss for encoders
                feat_e_dom_gan_loss_gg = - feat_e_dom_target_score
            else:
                raise ValueError("it_model_gan_update_rule must be 'wgan'.")

            # 1-2) domain feature - behavior adversarial loss
            if self.it_model_gan_update_rule == 'wgan':
                feat_d_cls_expert_score = tf.reduce_mean(se_d_cls_disc_logits)
                feat_d_cls_nonexp_score = tf.reduce_mean(tf.concat([sn_d_cls_disc_logits,
                                                                    tn_d_cls_disc_logits,], axis=0))
                feat_d_cls_gp_loss = self.gradient_penalty(
                    tf.concat([se_enc_d_out, se_enc_d_out, ], axis=0),
                    tf.concat([sn_enc_d_out, tn_enc_d_out, ], axis=0),
                    self._feat_d_cls_discriminator)
                # loss for discriminators
                feat_d_cls_gan_loss_dd = - feat_d_cls_expert_score + feat_d_cls_nonexp_score + self._gp_lambda * feat_d_cls_gp_loss
                # loss for encoders
                feat_d_cls_gan_loss_gg = - feat_d_cls_nonexp_score
            else:
                raise ValueError("it_model_gan_update_rule must be 'wgan'.")

            # 1-3) behavior feature - behavior prediction loss
            if self.it_model_gan_update_rule == 'wgan':
                feat_e_cls_expert_score = tf.reduce_mean(se_e_cls_disc_logits)
                feat_e_cls_nonexp_score = tf.reduce_mean(sn_e_cls_disc_logits)
                feat_e_cls_gp_loss = self.gradient_penalty(
                    se_enc_e_out,
                    sn_enc_e_out,
                    self._feat_e_cls_discriminator)
                # loss for discriminators
                feat_e_cls_gan_loss_dd = - feat_e_cls_expert_score + feat_e_cls_nonexp_score + self._gp_lambda * feat_e_cls_gp_loss
                # loss for encoders
                feat_e_cls_gan_loss_gg = feat_e_cls_gan_loss_dd
            else:
                raise ValueError("it_model_gan_update_rule must be 'wgan'.")

            # 1-4) domain prediction - domain prediction loss
            if self.it_model_gan_update_rule == 'wgan':
                feat_d_dom_source_score = tf.reduce_mean(tf.concat([se_d_dom_disc_logits,
                                                                    sn_d_dom_disc_logits, ], axis=0))
                feat_d_dom_target_score = tf.reduce_mean(tf.concat([tn_d_dom_disc_logits,
                                                                    tl_d_dom_disc_logits, ], axis=0))
                feat_d_dom_gp_loss = self.gradient_penalty(
                    tf.concat([se_enc_d_out, sn_enc_d_out, ], axis=0),
                    tf.concat([tl_enc_d_out, tn_enc_d_out, ], axis=0),
                    self._feat_d_cls_discriminator)
                # loss for discriminators
                feat_d_dom_gan_loss_dd = - feat_d_dom_source_score + feat_d_dom_target_score + self._gp_lambda * feat_d_dom_gp_loss
                # loss for encoders
                feat_d_dom_gan_loss_gg = feat_d_dom_gan_loss_dd
            else:
                raise ValueError("it_model_gan_update_rule must be 'wgan'.")

            # 2-1) source domain images, image adversarial loss
            if self.it_model_gan_update_rule == 'wgan':
                source_real_scores = tf.reduce_mean(se_trans_disc_logits) \
                                     + tf.reduce_mean(sn_trans_disc_logits)
                source_fake_scores = tf.reduce_mean(sn0se_trans_disc_logits) \
                                     + tf.reduce_mean(tf.concat([se0sn_trans_disc_logits,
                                                                 se0tn_trans_disc_logits,
                                                                 sn0tn_trans_disc_logits,
                                                                 ], axis=0))
                source_trans_gp_loss = self.gradient_penalty(
                                           tf.convert_to_tensor(se_ims, dtype=tf.float32),
                                           sn0se_trans_out, self._source_trans_discriminator) + \
                                       self.gradient_penalty(
                                           tf.concat([tf.convert_to_tensor(sn_ims, dtype=tf.float32),
                                                      tf.convert_to_tensor(sn_ims, dtype=tf.float32),
                                                      tf.convert_to_tensor(sn_ims, dtype=tf.float32), ], axis=0),
                                           tf.concat([se0sn_trans_out,
                                                      se0tn_trans_out,
                                                      sn0tn_trans_out], axis=0), self._source_trans_discriminator)
                # loss for discriminators
                source_trans_gan_loss_dd = - source_real_scores + source_fake_scores + self._gp_lambda * source_trans_gp_loss
                # loss for encoders/generators
                source_trans_gan_loss_gg = - source_fake_scores
            else:
                raise ValueError("it_model_gan_update_rule must be 'wgan'.")

            # 2-2) target domain images, image adversarial loss
            if self.it_model_gan_update_rule == 'wgan':
                target_real_scores = tf.reduce_mean(tn_trans_disc_logits) \
                                     + tf.reduce_mean(tl_trans_disc_logits)
                target_fake_scores = tf.reduce_mean(tf.concat([tn0sn_trans_disc_logits,
                                                               tl0sn_trans_disc_logits,
                                                               tl0tn_trans_disc_logits,
                                                               ], axis=0)) \
                                     + tf.reduce_mean(tn0tl_trans_disc_logits)
                target_trans_gp_loss = self.gradient_penalty(
                    tf.concat([tf.convert_to_tensor(tn_ims, dtype=tf.float32),
                               tf.convert_to_tensor(tn_ims, dtype=tf.float32),
                               tf.convert_to_tensor(tn_ims, dtype=tf.float32)], axis=0),
                    tf.concat([tn0sn_trans_out,
                               tl0sn_trans_out,
                               tl0tn_trans_out], axis=0), self._target_trans_discriminator) \
                                       + self.gradient_penalty(
                    tf.convert_to_tensor(tl_ims, dtype=tf.float32),
                    tn0tl_trans_out, self._target_trans_discriminator)
                # loss for discriminators
                target_trans_gan_loss_dd = - target_real_scores + target_fake_scores + self._gp_lambda * target_trans_gp_loss
                # loss for encoders/generators
                target_trans_gan_loss_gg = - target_fake_scores
            else:
                raise ValueError("it_model_gan_update_rule must be 'wgan'.")

            # 3) image reconstruction loss
            if self.type_recon_loss == 'l2':
                recon_loss = tf.reduce_mean(tf.square(
                    tf.concat([se_ims, sn_ims, tn_ims, tl_ims], axis=0) -
                    tf.concat([se0se_recon_out, sn0sn_recon_out, tn0tn_recon_out, tl0tl_recon_out], axis=0)),
                    )
            else:
                raise ValueError("type_recon_loss must be l2.")

            # 4) image cycle consistency loss
            if self.type_recon_loss == 'l2':
                cycle_loss = tf.reduce_mean(tf.square(
                    tf.concat([se_ims, se_ims, se_ims,
                               sn_ims, sn_ims, sn_ims,
                               tn_ims, tn_ims, tn_ims,
                               tl_ims, tl_ims, tl_ims], axis=0) -
                    tf.concat([
                        se0sn_sn0se_cycle_out, se0tn_tn0se_cycle_out, se0tl_tl0se_cycle_out,
                        sn0se_se0sn_cycle_out, sn0tn_tn0sn_cycle_out, sn0tl_tl0sn_cycle_out,
                        tn0se_se0tn_cycle_out, tn0sn_sn0tn_cycle_out, tn0tl_tl0tn_cycle_out,
                        tl0se_se0tl_cycle_out, tl0sn_sn0tl_cycle_out, tl0tn_tn0tl_cycle_out], axis=0)),
                    )  # L2 loss
            else:
                raise ValueError("type_recon_loss must be l2.")

            # 5) feature similarity loss
            if self.type_recon_loss == 'l2':
                enc_d_mean_loss = tf.reduce_mean(tf.square(
                    tf.reduce_mean(tf.concat([se_enc_d_out, tn_enc_d_out], axis=0), axis=0)
                    - tf.reduce_mean(tf.concat([sn_enc_d_out, tl_enc_d_out], axis=0), axis=0)))
                enc_e_mean_loss = tf.reduce_mean(tf.square(
                    tf.reduce_mean(sn_enc_e_out, axis=0)
                    - tf.reduce_mean(tn_enc_e_out, axis=0)))
            else:
                raise ValueError("type_recon_loss must be l2.")

            # 6) feature reconstruction loss
            if self.type_recon_loss == 'l2':
                enc_d_recon_loss = tf.reduce_mean(tf.square(
                    tf.concat([se_enc_d_out, sn_enc_d_out,
                               tn_enc_d_out, tl_enc_d_out], axis=0) -
                    tf.concat([se0se_recon_enc_d_out, sn0sn_recon_enc_d_out,
                               tn0tn_recon_enc_d_out, tl0tl_recon_enc_d_out], axis=0)),
                    )  # L2 loss
                enc_e_recon_loss = tf.reduce_mean(tf.square(
                    tf.concat([se_enc_e_out, sn_enc_e_out,
                               tn_enc_e_out, tl_enc_e_out], axis=0) -
                    tf.concat([se0se_recon_enc_e_out, sn0sn_recon_enc_e_out,
                               tn0tn_recon_enc_e_out, tl0tl_recon_enc_e_out], axis=0)),
                    )  # L2 loss
            else:
                raise ValueError("type_recon_loss must be l2.")

            # 7) feature regularization loss
            if self.c_norm_de == 0:
                se_enc_d_reg_loss = self.l2_regularize(se_enc_d_out)
                sn_enc_d_reg_loss = self.l2_regularize(sn_enc_d_out)
                tn_enc_d_reg_loss = self.l2_regularize(tn_enc_d_out)
                tl_enc_d_reg_loss = self.l2_regularize(tl_enc_d_out)
                se_enc_e_reg_loss = self.l2_regularize(se_enc_e_out)
                sn_enc_e_reg_loss = self.l2_regularize(sn_enc_e_out)
                tn_enc_e_reg_loss = self.l2_regularize(tn_enc_e_out)
                tl_enc_e_reg_loss = self.l2_regularize(tl_enc_e_out)
            else:
                se_enc_d_reg_loss = self.l2_regularize_with_norm(se_enc_d_out, self.c_norm_de)
                sn_enc_d_reg_loss = self.l2_regularize_with_norm(sn_enc_d_out, self.c_norm_de)
                tn_enc_d_reg_loss = self.l2_regularize_with_norm(tn_enc_d_out, self.c_norm_de)
                tl_enc_d_reg_loss = self.l2_regularize_with_norm(tl_enc_d_out, self.c_norm_de)
                se_enc_e_reg_loss = self.l2_regularize_with_norm(se_enc_e_out, self.c_norm_be)
                sn_enc_e_reg_loss = self.l2_regularize_with_norm(sn_enc_e_out, self.c_norm_be)
                tn_enc_e_reg_loss = self.l2_regularize_with_norm(tn_enc_e_out, self.c_norm_be)
                tl_enc_e_reg_loss = self.l2_regularize_with_norm(tl_enc_e_out, self.c_norm_be)

            # 8) feature cycle consistency loss
            if self.type_recon_loss == 'l2':
                feat_d_cycle_loss = tf.reduce_mean(tf.square(
                        tf.concat([se0sn_trans_enc_d_out, sn0se_trans_enc_d_out,
                                   se0tn_trans_enc_d_out, tn0se_trans_enc_d_out,
                                   se0tl_trans_enc_d_out, tl0se_trans_enc_d_out,
                                   sn0tn_trans_enc_d_out, tn0sn_trans_enc_d_out,
                                   sn0tl_trans_enc_d_out, tl0sn_trans_enc_d_out,
                                   tn0tl_trans_enc_d_out, tl0tn_trans_enc_d_out], axis=0) -
                        tf.concat([se_enc_d_out, sn_enc_d_out,
                                   se_enc_d_out, tn_enc_d_out,
                                   se_enc_d_out, tl_enc_d_out,
                                   sn_enc_d_out, tn_enc_d_out,
                                   sn_enc_d_out, tl_enc_d_out,
                                   tn_enc_d_out, tl_enc_d_out], axis=0)))
                feat_e_cycle_loss = tf.reduce_mean(tf.square(
                        tf.concat([se0sn_trans_enc_e_out, sn0se_trans_enc_e_out,
                                   se0tn_trans_enc_e_out, tn0se_trans_enc_e_out,
                                   se0tl_trans_enc_e_out, tl0se_trans_enc_e_out,
                                   sn0tn_trans_enc_e_out, tn0sn_trans_enc_e_out,
                                   sn0tl_trans_enc_e_out, tl0sn_trans_enc_e_out,
                                   tn0tl_trans_enc_e_out, tl0tn_trans_enc_e_out], axis=0) -
                        tf.concat([sn_enc_e_out, se_enc_e_out,
                                   tn_enc_e_out, se_enc_e_out,
                                   tl_enc_e_out, se_enc_e_out,
                                   tn_enc_e_out, sn_enc_e_out,
                                   tl_enc_e_out, sn_enc_e_out,
                                   tl_enc_e_out, tn_enc_e_out], axis=0)))
            else:
                raise ValueError("type_recon_loss must be l2.")

            # ==================================================
            # Total loss
            loss_gan_feat_e_dd = feat_e_dom_gan_loss_dd     # behavior feature adversarial, discrimniator
            loss_gan_feat_e_gg = feat_e_dom_gan_loss_gg     # behavior feature adversarial, encoder/generator
            loss_gan_feat_e_dd2 = feat_e_cls_gan_loss_dd    # behavior feature prediction, discriminator
            loss_gan_feat_e_gg2 = feat_e_cls_gan_loss_gg    # behavior feature prediction, encoder/generator
            loss_gan_feat_d_dd = feat_d_cls_gan_loss_dd     # domain feature adversarial, discrimniator
            loss_gan_feat_d_gg = feat_d_cls_gan_loss_gg     # domain feature adversarial, encoder/generator
            loss_gan_feat_d_dd2 = feat_d_dom_gan_loss_dd    # domain feature prediction, discrimniator
            loss_gan_feat_d_gg2 = feat_d_dom_gan_loss_gg    # domain feature adversarial, encoder/generator

            loss_gan_trans_dd = source_trans_gan_loss_dd \
                                + target_trans_gan_loss_dd  # image adversarial, discriminator
            loss_gan_trans_gg = source_trans_gan_loss_gg \
                                + target_trans_gan_loss_gg  # image adversarial, encoder/generator
            loss_recon = recon_loss                         # image reconstruction
            loss_cycle = cycle_loss                         # image cycle consistency

            loss_feat_d_mean = enc_d_mean_loss              # domain feature similarity
            loss_feat_e_mean = enc_e_mean_loss              # behavior feature similarity
            loss_feat_d_recon = enc_d_recon_loss            # domain feature reconstruction
            loss_feat_e_recon = enc_e_recon_loss            # behavior feature reconstruction

            loss_enc_d_reg = se_enc_d_reg_loss + sn_enc_d_reg_loss + tn_enc_d_reg_loss + tl_enc_d_reg_loss  # domain feature regularization
            loss_enc_e_reg = se_enc_e_reg_loss + sn_enc_e_reg_loss + tn_enc_e_reg_loss + tl_enc_e_reg_loss  # behavior feature regularization

            loss_feat_d_cycle = feat_d_cycle_loss           # domain feature cycle consistency
            loss_feat_e_cycle = feat_e_cycle_loss           # behavior feature cycle consistency

            if epoch == 0 or (epoch + 1) % self.eg_update_interval == 0:
                if self.it_model_gan_update_rule == 'wgan':
                    total_loss_disc_trans = 5.0 * self.c_gan_trans * loss_gan_trans_dd
                    total_loss_disc_feat = 5.0 * self.c_gan_feat * loss_gan_feat_e_dd \
                                           + 5.0 * self.c_gan_feat * loss_gan_feat_d_dd  \
                                           + 5.0 * self.c_gan_feat * loss_gan_feat_e_dd2 \
                                           + 5.0 * self.c_gan_feat * loss_gan_feat_d_dd2
                    total_loss_enc_e = self.c_gan_trans * loss_gan_trans_gg \
                                       + self.c_gan_feat * loss_gan_feat_e_gg \
                                       + self.c_recon * loss_recon \
                                       + self.c_cycle * loss_cycle \
                                       + self.c_feat_recon * loss_feat_e_recon \
                                       + self.c_feat_mean * loss_feat_e_mean \
                                       + self.c_feat_reg * loss_enc_e_reg \
                                       + self.c_gan_feat * loss_gan_feat_e_gg2 \
                                       + self.c_feat_cycle * loss_feat_e_cycle
                    total_loss_enc_d = self.c_gan_trans * loss_gan_trans_gg \
                                       + self.c_gan_feat * loss_gan_feat_d_gg \
                                       + self.c_recon * loss_recon \
                                       + self.c_cycle * loss_cycle \
                                       + self.c_feat_recon * loss_feat_d_recon \
                                       + self.c_feat_mean * loss_feat_d_mean \
                                       + self.c_feat_reg * loss_enc_d_reg \
                                       + self.c_gan_feat * loss_gan_feat_d_gg2 \
                                       + self.c_feat_cycle * loss_feat_d_cycle
                    total_loss_gen = self.c_gan_trans * loss_gan_trans_gg \
                                     + self.c_recon * loss_recon \
                                     + self.c_cycle * loss_cycle \
                                     + self.c_feat_recon * loss_feat_e_recon \
                                     + self.c_feat_recon * loss_feat_d_recon \
                                     + self.c_feat_cycle * loss_feat_d_cycle \
                                     + self.c_feat_cycle * loss_feat_e_cycle
                else:
                    raise ValueError("it_model_gan_update_rule must be 'wgan'.")
            else:
                total_loss_disc_trans = self.c_gan_trans * loss_gan_trans_dd
                total_loss_disc_feat = self.c_gan_feat * loss_gan_feat_e_dd \
                                       + self.c_gan_feat * loss_gan_feat_d_dd
                total_loss_enc_e = None
                total_loss_enc_d = None
                total_loss_gen = None

        # ==================================================
        # Compute gradients
        if total_loss_disc_trans is not None:
            gradients_disc_trans = tape.gradient(total_loss_disc_trans,
                                                  self._source_trans_discriminator.trainable_weights
                                                  + self._target_trans_discriminator.trainable_weights)
            gradients_disc_trans_norm = tf.linalg.global_norm(gradients_disc_trans)

        if total_loss_disc_feat is not None:
            gradients_disc_feat = tape.gradient(total_loss_disc_feat,
                                                self._feat_e_dom_discriminator.trainable_weights
                                                + self._feat_d_cls_discriminator.trainable_weights)
            gradients_disc_feat_norm = tf.linalg.global_norm(gradients_disc_feat)

        if total_loss_enc_e is not None:
            gradients_enc_e = tape.gradient(total_loss_enc_e,
                                          self._source_encoder_e.trainable_weights
                                          + self._target_encoder_e.trainable_weights)
            gradients_enc_e_norm = tf.linalg.global_norm(gradients_enc_e)

        if total_loss_enc_d is not None:
            gradients_enc_d = tape.gradient(total_loss_enc_d,
                                          self._source_encoder_d.trainable_weights
                                          + self._target_encoder_d.trainable_weights)
            gradients_enc_d_norm = tf.linalg.global_norm(gradients_enc_d)

        if total_loss_gen is not None:
            gradients_gen = tape.gradient(total_loss_gen,
                                          self._source_generator.trainable_weights
                                          + self._target_generator.trainable_weights)
            gradients_gen_norm = tf.linalg.global_norm(gradients_gen)

        # Clip gradients (if necessary)
        if self.it_max_grad_norm is not None:
            if total_loss_disc_trans is not None:
                gradients_disc_trans, _ = tf.clip_by_global_norm(gradients_disc_trans, self.it_max_grad_norm)
            if total_loss_disc_feat is not None:
                gradients_disc_feat, _ = tf.clip_by_global_norm(gradients_disc_feat, self.it_max_grad_norm)
            if total_loss_enc_e is not None:
                gradients_enc_e, _ = tf.clip_by_global_norm(gradients_enc_e, self.it_max_grad_norm)
            if total_loss_enc_d is not None:
                gradients_enc_d, _ = tf.clip_by_global_norm(gradients_enc_d, self.it_max_grad_norm)
            if total_loss_gen is not None:
                gradients_gen, _ = tf.clip_by_global_norm(gradients_gen, self.it_max_grad_norm)

        # Apply gradients
        if total_loss_disc_trans is not None:
            self._it_optimizer.apply_gradients(zip(gradients_disc_trans,
                                                   self._source_trans_discriminator.trainable_weights
                                                   + self._target_trans_discriminator.trainable_weights))

        if total_loss_disc_feat is not None:
            self._it_optimizer.apply_gradients(zip(gradients_disc_feat,
                                                self._feat_e_dom_discriminator.trainable_weights
                                                + self._feat_d_cls_discriminator.trainable_weights))

        if total_loss_enc_e is not None:
            self._it_optimizer.apply_gradients(zip(gradients_enc_e,
                                                self._source_encoder_e.trainable_weights
                                                + self._target_encoder_e.trainable_weights))

        if total_loss_enc_d is not None:
            self._it_optimizer.apply_gradients(zip(gradients_enc_d,
                                                self._source_encoder_d.trainable_weights
                                                + self._target_encoder_d.trainable_weights))

        if total_loss_gen is not None:
            self._it_optimizer.apply_gradients(zip(gradients_gen,
                                                self._source_generator.trainable_weights
                                                + self._target_generator.trainable_weights))

        # ==================================================
        # output results
        out = dict()
        if total_loss_disc_trans is not None:
            out['total_loss_disc_trans'] = total_loss_disc_trans.numpy()
            out['g_norm_disc_trans'] = gradients_disc_trans_norm.numpy()
            self.check_nan(total_loss_disc_trans)
            self.check_nan(gradients_disc_trans_norm)

        if total_loss_disc_feat is not None:
            out['total_loss_disc_feat'] = total_loss_disc_feat.numpy()
            out['g_norm_disc_feat'] = gradients_disc_feat_norm.numpy()
            self.check_nan(total_loss_disc_feat)
            self.check_nan(gradients_disc_feat_norm)

        if total_loss_enc_e is not None:
            out['total_loss_enc_e'] = total_loss_enc_e.numpy()
            out['g_norm_enc_e'] = gradients_enc_e_norm.numpy()
            self.check_nan(total_loss_enc_e)
            self.check_nan(gradients_enc_e_norm)

        if total_loss_enc_d is not None:
            out['total_loss_enc_d'] = total_loss_enc_d.numpy()
            out['g_norm_enc_d'] = gradients_enc_d_norm.numpy()
            self.check_nan(total_loss_enc_d)
            self.check_nan(gradients_enc_d_norm)

        if total_loss_gen is not None:
            out['total_loss_gen'] = total_loss_gen.numpy()
            out['g_norm_gen'] = gradients_gen_norm.numpy()
            self.check_nan(total_loss_gen)
            self.check_nan(gradients_gen_norm)

        out['loss_gan_trans_dd'] = loss_gan_trans_dd.numpy()
        out['loss_gan_trans_gg'] = loss_gan_trans_gg.numpy()
        self.check_nan(loss_gan_trans_dd)
        self.check_nan(loss_gan_trans_gg)

        out['loss_gan_feat_e_dd'] = loss_gan_feat_e_dd.numpy()
        out['loss_gan_feat_e_gg'] = loss_gan_feat_e_gg.numpy()
        out['loss_gan_feat_d_dd'] = loss_gan_feat_d_dd.numpy()
        out['loss_gan_feat_d_gg'] = loss_gan_feat_d_gg.numpy()
        out['loss_gan_feat_e_dd2'] = loss_gan_feat_e_dd2.numpy()
        out['loss_gan_feat_e_gg2'] = loss_gan_feat_e_gg2.numpy()
        out['loss_gan_feat_d_dd2'] = loss_gan_feat_d_dd2.numpy()
        out['loss_gan_feat_d_gg2'] = loss_gan_feat_d_gg2.numpy()
        self.check_nan(loss_gan_feat_e_dd)
        self.check_nan(loss_gan_feat_e_gg)
        self.check_nan(loss_gan_feat_d_dd)
        self.check_nan(loss_gan_feat_d_gg)
        self.check_nan(loss_gan_feat_e_dd2)
        self.check_nan(loss_gan_feat_e_gg2)
        self.check_nan(loss_gan_feat_d_dd2)
        self.check_nan(loss_gan_feat_d_gg2)

        out['loss_recon'] = loss_recon.numpy()
        self.check_nan(loss_recon)

        out['loss_cycle'] = loss_cycle.numpy()
        self.check_nan(loss_cycle)

        out['loss_feat_d_mean'] = loss_feat_d_mean.numpy()
        out['loss_feat_e_mean'] = loss_feat_e_mean.numpy()
        self.check_nan(loss_feat_d_mean)
        self.check_nan(loss_feat_e_mean)

        out['loss_feat_d_recon'] = loss_feat_d_recon.numpy()
        out['loss_feat_e_recon'] = loss_feat_e_recon.numpy()
        self.check_nan(loss_feat_d_recon)
        self.check_nan(loss_feat_e_recon)

        out['reg_se_enc_d'] = se_enc_d_reg_loss.numpy()
        out['reg_sn_enc_d'] = sn_enc_d_reg_loss.numpy()
        out['reg_tn_enc_d'] = tn_enc_d_reg_loss.numpy()
        out['reg_tl_enc_d'] = tl_enc_d_reg_loss.numpy()
        out['reg_se_enc_e'] = se_enc_e_reg_loss.numpy()
        out['reg_sn_enc_e'] = sn_enc_e_reg_loss.numpy()
        out['reg_tn_enc_e'] = tn_enc_e_reg_loss.numpy()
        out['reg_tl_enc_e'] = tl_enc_e_reg_loss.numpy()

        out['loss_feat_d_cycle'] = feat_d_cycle_loss.numpy()
        out['loss_feat_e_cycle'] = feat_e_cycle_loss.numpy()

        del tape
        return out

    # Misc.
    # ==================================================
    @tf.function
    def call(self, se_inputs, sn_inputs, tn_inputs, tl_inputs, agent_inputs):
        """
        This method will be called only once to build the model.
        """
        self.image_translation(se_inputs, sn_inputs, tn_inputs, tl_inputs,
                               tf.ones_like(se_inputs), tf.ones_like(sn_inputs), tf.ones_like(tn_inputs), tf.ones_like(tl_inputs))
        if self._expert_discriminator is not None:
            self.expert_discrimination(se_inputs, tl_inputs, sn_inputs, tn_inputs, tf.ones_like(tl_inputs))
        if self._agent is not None:
            self._agent(agent_inputs)

    def reshape_input_images(self, input_pre):
        """
        Reshape input with shape (B, 4, W, H, C) to (B, W, H, C * 4).
        """
        if input_pre.shape[-1] == 3:
            if self._past_frames == 4:
                input_post = tf.concat([input_pre[:, 0, :, :, :], input_pre[:, 1, :, :, :],
                                        input_pre[:, 2, :, :, :], input_pre[:, 3, :, :, :]], axis=-1)
            elif self._past_frames == 2:
                input_post = tf.concat([input_pre[:, 0, :, :, :], input_pre[:, 1, :, :, :]], axis=-1)
            else:
                raise NotImplementedError
        elif (input_pre.shape[-1] == 12 and self._past_frames == 4) or (input_pre.shape[-1] == 6 and self._past_frames == 2):
            input_post = input_pre
        else:
            raise AssertionError("Invalid input shape")
        return input_post

    @staticmethod
    def get_masks(images):
        """
        Obtain masks for images (0.0 if an image is black, and 1.0 otherwise)
        """
        if tf.is_tensor(images):
            _masks = np.ones_like(images.numpy())
        else:
            _masks = np.ones_like(images)
        min_val = 0.5 / 256 * images.shape[-3] * images.shape[-2] * 3
        for i in range(images.shape[0]):
            if tf.reduce_sum(images[i, :, :, -6:-3]) <= min_val + 1e-7:
                _masks[i, :, :, -6:-3] = 0.
            if tf.reduce_sum(images[i, :, :, -3:]) <= min_val + 1e-7:
                _masks[i, :, :, -3:] = 0.
        if tf.is_tensor(images):
            return tf.constant(_masks)
        else:
            return _masks

    @staticmethod
    def gradient_penalty(real, fake, discriminator):
        """
        Compute gradient penalty (for WGAN-GP).
        """
        with tf.GradientTape() as tape_temp:
            tape_temp.watch([real, fake])
            if len(real.shape) == 4:
                bs, w, h, c = real.shape
                alpha = tf.reshape(tf.tile(tf.random.uniform([1]), (bs * w * h * c, )), [bs, w, h, c])
                interpolated = alpha * real + (1. - alpha) * fake
                interpolated_score = discriminator(interpolated)
                gradients = tape_temp.gradient(interpolated_score, [interpolated])[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]) + 1e-7)
                grad_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            else:
                bs, ll = real.shape
                alpha = tf.reshape(tf.tile(tf.random.uniform([1]), (bs * ll,)), [bs, ll])
                _interpolated = alpha * real + (1. - alpha) * fake
                interpolated = _interpolated
                interpolated_score = discriminator(interpolated)
                gradients = tape_temp.gradient(interpolated_score, [interpolated])[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]) + 1e-7)
                grad_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        try:
            del tape_temp
        except NameError:
            pass
        return grad_penalty

    @staticmethod
    def check_nan(x):
        """
        Check if a tensor has the 'Nan' value.
        """
        if tf.math.is_nan(x):
            print("nan is occurred.")
            import pdb; pdb.set_trace()
            pass

    @staticmethod
    def l2_regularize(x):
        """
        Return l2 norm of a tensor.
        """
        return tf.reduce_mean(tf.square(x))

    @staticmethod
    def l2_regularize_with_norm(x, c):
        """
        Return l2 norm of a tensor.
        """
        return tf.reduce_mean(tf.square(tf.norm(x, axis=1) - c))

    @staticmethod
    def mean(x):
        """
        Return l2 norm of a tensor.
        """
        return tf.reduce_mean(x, axis=0)


    # Deprecated
    # ==================================================
    def summary_model(self, _inputs):
        """
        Show the image translation model structure.
        """

        self.summary()

        inputs = self.reshape_input_images(_inputs)

        print("\033[96m")
        # ==================================================
        print("Encoder - d")
        print("-" * 100)
        feats_d = inputs[:, :, :, 0:3]
        print("{:70s}\t{}".format("Input shape (sample)", feats_d.shape))
        for layer in self._source_encoder_d._layers:
            feats_d = layer(feats_d)
            print("{:70s}\t{}".format(str(layer.__class__), feats_d.shape))
        print("{:70s}\t{}".format("Output shape (sample)", feats_d.shape))
        print()

        # ==================================================
        print("Encoder - e")
        print("-" * 100)
        print("{:70s}\t{}".format("Input shape (sample)", inputs.shape))
        feats_e = inputs
        for layer in self._source_encoder_e._layers:
            feats_e = layer(feats_e)
            print("{:70s}\t{}".format(str(layer.__class__), feats_e.shape))
        print("{:70s}\t{}".format("Output shape (sample)", feats_e.shape))
        print()

        # ==================================================
        print("Generator")
        print("-" * 100)
        print("{:70s}\t(d) {}, (e) {}".format("Input shape (sample)", feats_d.shape, feats_e.shape))
        outputs = feats_e
        im_size = self._source_generator.get_im_size(outputs.shape)
        outputs = tf.reshape(outputs, [1, im_size, im_size, self._source_generator._n_input_channels])
        feats_d = tf.reshape(feats_d, [1, -1])
        print("{:70s}\t{}".format("tf.reshape", outputs.shape))
        for layer in self._source_generator._layers:
            im_size = outputs.shape[1]
            outputs_d = tf.reshape(tf.tile(feats_d, (1, im_size ** 2)),
                                   [outputs.shape[0], im_size, im_size, feats_d.shape[-1]])
            outputs = tf.concat([outputs_d, outputs], axis=-1)
            outputs = layer(outputs)
            print("{:70s}\t{}".format(str(layer.__class__), outputs.shape))
        print("{:70s}\t{}".format("Output shape (sample)", outputs.shape))
        print()

        # ==================================================
        print("Discriminator (domain independence)")
        print("-" * 100)
        print("{:70s}\t{}".format("Input shape (sample)", feats_e.shape))
        outputs = feats_e
        for layer in self._feat_e_dom_discriminator._dis_layers:
            outputs = layer(outputs)
            print("{:70s}\t{}".format(str(layer.__class__), outputs.shape))
        print("{:70s}\t{}".format("Output shape (sample)", outputs.shape))
        print()

        # ==================================================
        print("Discriminator (class independence)")
        print("-" * 100)
        print("{:70s}\t{}".format("Input shape (sample)", feats_d.shape))
        outputs = feats_d
        for layer in self._feat_d_cls_discriminator._dis_layers:
            outputs = layer(outputs)
            print("{:70s}\t{}".format(str(layer.__class__), outputs.shape))
        print("{:70s}\t{}".format("Output shape (sample)", outputs.shape))
        print()

        # ==================================================
        print("Discriminator (translated image)")
        print("-" * 100)
        print("{:70s}\t{}".format("Input shape (sample)", inputs.shape))
        outputs = inputs
        for layer in self._source_trans_discriminator._dis_layers:
            outputs = layer(outputs)
            print("{:70s}\t{}".format(str(layer.__class__), outputs.shape))
        print("{:70s}\t{}".format("Output shape (sample)", outputs.shape))
        print()

        # ==================================================
        print("Loss coefficients")
        print("-" * 100)
        print("c_gan_trans        : {}".format(self.c_gan_trans))
        print("c_gan_feat         : {}".format(self.c_gan_feat))
        print("c_recon            : {}".format(self.c_recon))
        print("c_cycle            : {}".format(self.c_cycle))
        print("c_feat_mean        : {}".format(self.c_feat_mean))
        print("c_feat_recon       : {}".format(self.c_feat_recon))
        print("c_feat_cycle       : {}".format(self.c_feat_cycle))
        print("type_recon_loss    : {}".format(self.type_recon_loss))
        print("eg_update_interval : {}".format(self.eg_update_interval))
        print("it_max_grad_norm   : {}".format(self.it_max_grad_norm))
        print("it_learning_rate   : {}".format(self._it_lr))
        print()

        # ==================================================
        print("\033[0m\n")


# ==================================================
# ==================================================
class D3ILModelwithPolicy(D3ILModel):
    """
    Imitation - Type 1D

    Expert discriminator * 1 (expert vs. non-expert)

    (1 - image translation)
    tn_input -> (target_encoder_d) -> tn_enc_d_out -|
                                                    |-> (target_generator) -> tn0se_trans_out
    se_input -> (source_encoder_e) -> se_enc_e_out -|

    (2 - expert discrimination)
    [Expert] tn0se_trans_out -> (target_encoder_e) -> tn0se_enc_e_out -|
                                                                       |-> (expert_discriminator) -> output
    [Nonexp]        tl_input -> (target_encoder_e) ->    tl_enc_e_out -|
    """
    def __init__(self, agent,
                 make_encoder_d_fn,
                 make_encoder_e_fn,
                 make_generator_fn,
                 make_dom_disc_fn,
                 make_cls_disc_fn,
                 make_trans_disc_fn,
                 make_expert_disc_fn,
                 c_gan_trans=1.0,
                 c_gan_feat=1.0,
                 c_recon=1.0,
                 c_cycle=1.0,
                 c_feat_mean=1.0,
                 c_feat_recon=1.0,
                 c_feat_reg=0.0,
                 c_feat_cycle=1.0,
                 c_norm_de=1,
                 c_norm_be=1,
                 type_recon_loss='l2',
                 eg_update_interval=1,
                 it_max_grad_norm=None,
                 it_lr=0.0003,
                 d_rew='mixed',
                 d_max_grad_norm=None,
                 d_learning_rate=0.0003,
                 past_frames=4,
                 optimizer=None):
        super(D3ILModelwithPolicy, self).__init__(agent,
                                              make_encoder_d_fn,
                                              make_encoder_e_fn,
                                              make_generator_fn,
                                              make_dom_disc_fn,
                                              make_cls_disc_fn,
                                              make_trans_disc_fn,
                                              make_expert_disc_fn,
                                              c_gan_trans,
                                              c_gan_feat,
                                              c_recon,
                                              c_cycle,
                                              c_feat_mean,
                                              c_feat_recon,
                                              c_feat_reg,
                                              c_feat_cycle,
                                              c_norm_de,
                                              c_norm_be,
                                              type_recon_loss,
                                              eg_update_interval,
                                              it_max_grad_norm,
                                              it_lr,
                                              d_rew,
                                              d_max_grad_norm,
                                              d_learning_rate,
                                              past_frames,
                                              optimizer)
        # Discrminator (real)
        self._expert_discriminator = make_expert_disc_fn()
        self._debug_list = deque(maxlen=10000)

    # Phase 2
    # ==================================================
    def expert_discrimination(self, se_inputs, tl_inputs, sn_inputs, tn_inputs, masks=None):
        """
        Returns the output of expert discriminator (and a translated image)
        """
        # mask
        if masks is None:
            masks = self.get_masks(se_inputs)

        # feature extraction
        tn_enc_d_out = self._target_encoder_d(tn_inputs[:, :, :, 0:3])
        se_enc_e_out = self._source_encoder_e(se_inputs)

        # translation
        tn0se_trans_out = self._target_generator(tn_enc_d_out, se_enc_e_out) * masks + (1.0 - masks) * 0.5 / 256  # tn(t) + se(e)

        # feature extraction
        tn0se_enc_e_out = self._target_encoder_e(tn0se_trans_out)
        tl_enc_d_out = self._target_encoder_e(tl_inputs)

        # discrimination
        tn0se_expert_disc_logits = self._expert_discriminator(tf.stop_gradient(tn0se_enc_e_out))
        tl_expert_disc_logits = self._expert_discriminator(tl_enc_d_out)
        return tn0se_expert_disc_logits, tl_expert_disc_logits, tn0se_enc_e_out, tl_enc_d_out, tn0se_trans_out

    def train(self, se_buffer, sn_buffer, tn_buffer, agent_buffer, l_batch_size, l_updates, l_act_delay,
              d_updates, d_batch_size, it_updates, it_batch_size, epoch, pretrain_epochs, nn_updates, step_counter, save_final_path=None):
        """
        This method includes the proposed update rule for both the expert discriminator and the policy.
        It may also include the update rule for image translation model.
        """
        # ==================================================
        # Train IT (image translation model)
        for i in range(it_updates):
            # Get minibatch (shape = (batch_size, 4, W, H, 3))
            se_ims = self.reshape_input_images(se_buffer.get_random_batch(it_batch_size)['ims'])
            sn_ims = self.reshape_input_images(sn_buffer.get_random_batch(it_batch_size)['ims'])
            tn_ims = self.reshape_input_images(tn_buffer.get_random_batch(it_batch_size)['ims'])
            tl_ims = self.reshape_input_images(agent_buffer.get_random_batch(it_batch_size, False)['ims'])

            # Train model
            out = self.train_image_translation(se_ims, sn_ims, tn_ims, tl_ims,
                                               pretrain_epochs + nn_updates * it_updates + i)

        # ==================================================
        # Train D (expert_discriminator)
        for i in range(d_updates):
            se_ims = self.reshape_input_images(se_buffer.get_random_batch(d_batch_size)['ims'])
            tn_ims = self.reshape_input_images(tn_buffer.get_random_batch(d_batch_size)['ims'])
            tl_ims = tf.concat([self.reshape_input_images(tn_buffer.get_random_batch(d_batch_size // 2)['ims']),
                                self.reshape_input_images(
                                    agent_buffer.get_random_batch(d_batch_size // 2, False)['ims'])], axis=0)

            with tf.GradientTape(persistent=True) as tape:
                tn0se_expert_disc_logits, tl_expert_disc_logits, tn0se_enc_e_out, tl_enc_d_out, tn0se_trans_out \
                    = self.expert_discrimination(se_ims, tl_ims, None, tn_ims)

                if self.expert_d_update_rule == 'gan':
                    # Normal GAN style
                    expert_label = tf.ones_like(tn0se_expert_disc_logits)
                    nonexp_label = tf.zeros_like(tl_expert_disc_logits)
                    expert_gp_loss = self.gradient_penalty(
                        tn0se_enc_e_out,
                        tl_enc_d_out,
                        self._expert_discriminator)
                    expert_gan_loss_dd = tf.reduce_mean(tf.losses.binary_crossentropy(
                        tf.concat([expert_label,
                                   nonexp_label], axis=0),
                        tf.concat([tf.math.sigmoid(tn0se_expert_disc_logits) + self._epsilon,
                                   tf.math.sigmoid(tl_expert_disc_logits) + self._epsilon], axis=0))) \
                        + self._gp_lambda_reward * expert_gp_loss
                else:
                    raise ValueError("d_update_style must be 'gan'.")
                total_loss_disc_expert = expert_gan_loss_dd

            # Compute gradient
            gradients_disc_expert = tape.gradient(total_loss_disc_expert,
                                                  self._expert_discriminator.trainable_weights)
            gradients_disc_expert_norm = tf.linalg.global_norm(gradients_disc_expert)

            # Apply gradient
            self._d_optimizer.apply_gradients(zip(gradients_disc_expert,
                                              self._expert_discriminator.trainable_weights))
            del tape

        # ==================================================
        # Train G (policy)
        self._agent.train(agent_buffer, l_batch_size, l_updates, l_act_delay)
        if it_updates > 0:
            return out

    def get_reward(self, _inputs):
        """
        Compute reward for tl_inputs
        1D: tl_input -> (target_encoder_e) -> feat_e -> (expert_discriminator) -> reward
        """
        inputs = self.reshape_input_images(_inputs)
        feat_e = self._target_encoder_e(inputs)
        if self.expert_d_update_rule == 'gan':
            if self._rew == 'positive':
                rewards = -tf.math.log(1.0 - tf.math.sigmoid(self._expert_discriminator(feat_e)) + self._epsilon)
            elif self._rew == 'negative':
                rewards = tf.math.log(tf.math.sigmoid(self._expert_discriminator(feat_e)) + self._epsilon)
            elif self._rew == 'mixed':
                rewards = tf.math.log(tf.math.sigmoid(self._expert_discriminator(feat_e)) + self._epsilon) \
                          - tf.math.log(1.0 - tf.math.sigmoid(self._expert_discriminator(feat_e)) + self._epsilon)
            else:
                raise ValueError('Invalid self._rew')
        else:
            raise ValueError("d_update_style must be 'gan'.")
        return rewards
