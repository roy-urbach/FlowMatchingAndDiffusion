from utils import runs_on_batch, counter
import numpy as np


@runs_on_batch
def midpoint_method(xt, f, n, c, start_t=0, keep_middle=True, drop_c=0):
    xt = xt.copy()
    xts = [xt]
    h = 1 / n
    if c is not None:
        c0_vec = np.array([1] + [0] * (c.shape[-1] - 1))
    for tn in counter(np.linspace(start_t + h, 1, n)):
        if c is not None:
            cur_c = c.copy()
            if drop_c:
                drop_inds = np.random.rand(len(c)) < drop_c
                if len(c.shape) > 1:
                    cur_c[drop_inds] = c0_vec
                else:
                    cur_c[drop_inds] = 0
        else:
            cur_c = c

        tn = np.full(len(xt), tn)
        xt = xt + h * f(xt + h / 2 * f(xt, tn, c=cur_c), tn + h / 2, c=cur_c)
        if keep_middle:
            xts.append(xt)
    return np.stack(xts, axis=-1)


def flow_sampling(num_samples, shape, ut_theta, t_theta, c_theta=None, c=None, num_steps=10, keep_middle=True,
                  drop_c=0):
    def predict(xt, t, c=None):
        enc_t = t_theta(t)
        ut_theta_inp = [xt, enc_t]
        if c_theta is not None:
            if c is not None:
                ut_theta_inp.append(c_theta(c))
        ut_hat = ut_theta(np.concat(ut_theta_inp, axis=-1))
        return ut_hat

    noise = np.random.randn(*([num_samples] + list(shape)))
    return midpoint_method(noise, predict, num_steps, start_t=0, keep_middle=keep_middle, c=c, drop_c=drop_c)


# TODO: diffusion
