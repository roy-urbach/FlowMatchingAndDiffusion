import tensorflow as tf
from utils import expand_t


def flow_matching_loss(ut_theta, t_theta, x1, c_theta=None, c=None, at=None):
    n = tf.shape(x1)[0]
    t = tf.random.uniform([n])
    alpha_t = expand_t(at(t) if at is not None else t, x1.shape)
    x0 = tf.random.normal(x1.shape)
    xt = alpha_t * x0 + (1 - alpha_t) * x1
    ut = x1 - x0
    enc_t = t_theta(t)
    f_ut_inp = [xt, enc_t]
    if c_theta is not None:
        if c is not None:
            f_ut_inp.append(c_theta(c))
    ut_hat = ut_theta(tf.concat(f_ut_inp, axis=-1))
    return tf.reduce_mean((ut_hat - ut) ** 2)


# didn't examine yet
def diffusion_loss(eps_theta, t_theta, x0, c_theta=None, c=None, at=None, lambda_t=None):
    n = tf.shape(x0)[0]
    t = tf.random.uniform([n])
    alpha_t = expand_t(at(t) if at is not None else t, x0.shape)
    eps = tf.random.randn(x0.shape)
    xt = tf.math.sqrt(alpha_t) * x0 + tf.math.sqrt(1 - alpha_t) * eps
    enc_t = t_theta(t)
    wt = lambda_t(t) if lambda_t is not None else 1
    eps_theta_inp = [xt, enc_t]
    if c_theta is not None:
        if c is not None:
            eps_theta_inp.append(c_theta(c))
    eps_hat = eps_theta(tf.concat(eps_theta_inp, axis=-1))
    return tf.reduce_mean(wt * tf.reduce_sum((eps_hat - eps) ** 2, axis=list(range(len(x1.shape[1:])))))


# didn't examine yet
def diffusion_loss_predict_target(x0_theta, t_theta, x0, c_theta=None, c=None, at=None, lambda_t=None):
    n = tf.shape(x0)[0]
    t = tf.random.uniform([n])
    alpha_t = expand_t(at(t) if at is not None else t, x0.shape)
    eps = tf.random.randn(x0.shape)
    xt = tf.math.sqrt(alpha_t) * x0 + tf.math.sqrt(1 - alpha_t) * eps
    enc_t = t_theta(t)
    wt = lambda_t(t) if lambda_t is not None else 1
    x0_theta_inp = [xt, enc_t]
    if c_theta is not None:
        if c is not None:
            x0_theta_inp.append(c_theta(c))
    x0_hat = x0_theta(tf.concat(x0_theta_inp, axis=-1))
    return tf.reduce_mean(wt * tf.reduce_sum((x0_hat - x0) ** 2, axis=list(range(len(x0.shape[1:])))))
