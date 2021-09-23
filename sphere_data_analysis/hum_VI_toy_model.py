# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import psi as digamma
import os

# %%
def gen_hum(s, bg_hum=50, k1=0.25, k2=0.5, obs_noise_var=9.0, internal_noise_var=1.0):
    x = np.ones(len(s)+1) * bg_hum
    for n in range(len(s)):
        x[n+1] = x[n] + (1-s[n])*k1*(bg_hum - x[n]) + s[n]*k2*(100. - x[n]) + np.random.normal(loc=0, scale=np.sqrt(internal_noise_var))
    x = x + np.random.normal(loc=0, scale=np.sqrt(obs_noise_var), size=len(x))
    return x

# %%
true_s = np.zeros(100)
true_s[10:20] = 1.0
x = gen_hum(true_s)

# fig, ax = plt.subplots(1)
# ax.plot(x, label='humidity')
# ax.set_xlabel('time step'); ax.set_ylabel('humidity [%]')
# ax.set_title('simple humidity model')
# ax2 = ax.twinx()
# ax2.plot(true_s, color='C1', label='state')
# ax2.set_ylabel('state')
# ax2.legend()

# %%
def humidity_latent_event_detection(x):
    """
        Max-sum (Viterbi) algorithm 
    """
    bg_hum = 60.0 # true value 50.0
    k1 = 0.29   # true value 0.25
    k2 = 0.4    # true value 0.5

    # transition probability s1->s2 = T[s1,s2]
    T = np.array([[0.9, 0.1], 
                  [0.1, 0.9]]) 

    N = len(x)-1

    s = np.zeros((N, 2))
    s_in_x = np.zeros((N, 2))
    s_sel = np.zeros((N, 2))

    # prepare messages from x[] 
    for n in range(N):
        for s_ in [0, 1]:
            x_ = x[n] + (1-s_)*k1*(bg_hum - x[n]) + s_*k2*(100. - x[n])
            s_in_x[n, s_] = np.log(norm.pdf(x[n+1], loc=x_, scale=5))

    # message passing to root node (s[0])
    s_ = np.zeros(2)
    s_in_ = np.zeros(2) # message from the future s[n]
    for n in reversed(range(N)):
        s_[0] = s_in_[0] + s_in_x[n,0]
        s_[1] = s_in_[1] + s_in_x[n,1]
        s_in_[0] = np.max([T[0,0]+s_[0], T[0,1]+s_[1]])
        s_in_[1] = np.max([T[1,0]+s_[0], T[1,1]+s_[1]])
        s_sel[n,0] = np.argmax([T[0,0]+s_[0], T[0,1]+s_[1]])
        s_sel[n,1] = np.argmax([T[1,0]+s_[0], T[1,1]+s_[1]])
        
    # decode (trace selected path)
    s = np.zeros(N)
    s[0] = 0
    for n in range(N-1):
        s[n+1] = s_sel[n+1,int(s[n])]

    return s

# %%
class hum_VI:
    """
    Humidity event detection with VI
    """
    def __init__(self, x, x_bg, obs_var):
        self.x = x
        self.x_bg = x_bg
        self.N = len(x)

        # hypeter parameters
        self.la = 1/obs_var # observation noise precision

        # prior parameters
        self.pr_mu_k1 = 0.1  # k1 prior mean
        self.pr_la_k1 = 100. # k1 prior lambda (precision)
        self.pr_mu_k2 = 0.5  # k2 prior mean
        self.pr_la_k2 = 50.  # k2 prior lambda (precision)
        self.pr_al_T = np.array([[1000., 1.], [1.,100.]]) # T prior alpah (Dirichlet)

        # distribution variables
        self.mu_k1 = self.pr_mu_k1
        self.la_k1 = self.pr_la_k1
        self.mu_k2 = self.pr_mu_k2
        self.la_k2 = self.pr_la_k2
        self.alpha_T = self.pr_al_T
        self.alpha0_T = np.tile(np.sum(self.alpha_T, axis=1),(2,1)).T

        # distribution for the latent variable (for debug)
        self.r = np.nan
        return

    def update_model_parameters(self, p_y_0, p_y_1, Nji):
        # update k1 params
        self.la_k1 = self.la * np.sum((self.x_bg[:-1] - self.x[:-1])**2 * p_y_0) + self.pr_la_k1
        self.mu_k1 = self.la / self.la_k1 * np.sum((self.x_bg[:-1] - self.x[:-1]) * p_y_0 * (self.x[1:] - self.x[:-1]))
        # update k2 params
        self.la_k2 = self.la * np.sum((100 - self.x[:-1])**2 * p_y_1) + self.pr_la_k2
        self.mu_k2 = self.la / self.la_k2 * np.sum((100 - self.x[:-1]) * p_y_1 * (self.x[1:] - self.x[:-1]))
        # update T params
        self.alpha_T = self.pr_al_T + Nji
        self.alpha0_T = np.tile(np.sum(self.alpha_T, axis=1),(2,1)).T
        return

    def _f_factor(self, x_t, x_t1, x_bg):
        ret = np.zeros(2)
        ret[0] = -self.la / 2.0 * ((x_t1 - x_t)**2 + (1/self.la_k1 + self.mu_k1**2) * (x_bg - x_t)**2 - 2*(x_t1 - x_t) * self.mu_k1 * (x_bg - x_t))
        ret[1] = -self.la / 2.0 * ((x_t1 - x_t)**2 + (1/self.la_k2 + self.mu_k2**2) * (100  - x_t)**2 - 2*(x_t1 - x_t) * self.mu_k2 * (100  - x_t))
        return ret

    def _g_factor(self,):
        ElogT = digamma(self.alpha_T) - digamma(self.alpha0_T)
        return ElogT

    def _forward_message(self):
        log_f_msg = np.zeros((2, self.N-1))
        log_prev_ = np.zeros(2)
        g_fac = self._g_factor()
        for n in range(self.N-1):
            f_fac = self._f_factor(self.x[n], self.x[n+1], self.x_bg[n])
            log_f_msg[0,n] = np.logaddexp(g_fac[0,0]+log_prev_[0], g_fac[1,0]+log_prev_[1]) + f_fac[0]
            log_f_msg[1,n] = np.logaddexp(g_fac[0,1]+log_prev_[0], g_fac[1,1]+log_prev_[1]) + f_fac[1]
            log_prev_[0] = log_f_msg[0,n]
            log_prev_[1] = log_f_msg[1,n]
        return log_f_msg

    def _backward_message(self):
        log_b_msg = np.zeros((2, self.N-1))
        log_prev_ = np.zeros(2)
        g_fac = self._g_factor()
        for n in reversed(range(self.N-2)):
            f_fac = self._f_factor(self.x[n+1], self.x[n+2], self.x_bg[n+1])
            log_b_msg[0,n] = np.logaddexp(f_fac[0]+g_fac[0,0]+log_prev_[0], f_fac[1]+g_fac[0,1]+log_prev_[1])
            log_b_msg[1,n] = np.logaddexp(f_fac[0]+g_fac[1,0]+log_prev_[0], f_fac[1]+g_fac[1,1]+log_prev_[1])
            log_prev_[0] = log_b_msg[0,n]
            log_prev_[1] = log_b_msg[1,n]
        return log_b_msg

    def update_latent_variable(self,):
        for_msg = self._forward_message()
        bak_msg = self._backward_message()

        # compute marginal probability r[y,t] y=0/1, t=[0..N-1]
        log_r = for_msg + bak_msg
        log_r = log_r - np.logaddexp(log_r[0,:], log_r[1,:])
        self.r = np.exp(log_r)

        # compute expected number of transitions (N_ji)
        log_s = np.zeros((2,2,self.N-2))
        g_n = self._g_factor()
        for n in range(self.N-2):
            r_n = np.tile(for_msg[:,n], (2,1)).T
            b_n1 = np.tile(bak_msg[:,n+1], (2,1))
            f_n1 = np.tile(self._f_factor(self.x[n+1], self.x[n+2], self.x_bg[n+1]), (2,1))
            log_s_ = r_n + b_n1 + f_n1 + g_n
            log_s_ = log_s_ - np.tile( np.logaddexp(log_s_[:,0], log_s_[:,1]), (2,1) ).T
            log_s[:,:,n] = log_s_
        s = np.exp(log_s)
        N_ji = np.sum(s, axis=2)
        return self.r, s, N_ji

    def _viterbi_backward_message(self):
        log_b_msg = np.zeros((2, self.N-1))
        path_sel = np.zeros((2, self.N-1))
        log_prev_ = np.zeros(2)
        g_fac = self._g_factor()
        for n in reversed(range(self.N-2)):
            f_fac = self._f_factor(self.x[n+1], self.x[n+2], self.x_bg[n+1])
            log_b_msg[0,n] = np.max([f_fac[0]+g_fac[0,0]+log_prev_[0], f_fac[1]+g_fac[0,1]+log_prev_[1]])
            log_b_msg[1,n] = np.max([f_fac[0]+g_fac[1,0]+log_prev_[0], f_fac[1]+g_fac[1,1]+log_prev_[1]])
            log_prev_[0] = log_b_msg[0,n]
            log_prev_[1] = log_b_msg[1,n]
            path_sel[0,n] = np.argmax([f_fac[0]+g_fac[0,0]+log_prev_[0], f_fac[1]+g_fac[0,1]+log_prev_[1]])
            path_sel[1,n] = np.argmax([f_fac[0]+g_fac[1,0]+log_prev_[0], f_fac[1]+g_fac[1,1]+log_prev_[1]])
        return log_b_msg, path_sel

    def _viterbi_path_trace(self, path_sel, init_state=0):
        state = np.zeros(self.N-1)
        state[0] = init_state
        for n in range(self.N-2):
            state[n+1] = path_sel[int(state[n]), n]
        return state

    def process(self, n=20):
        # Variational Inference
        for n in range(n):
            print(f"iteration {n}")
            r,s,Nji = self.update_latent_variable()
            self.update_model_parameters(r[0,:], r[1,:], Nji)

        # Viterbi decoding
        _, path_sel = self._viterbi_backward_message()
        state = self._viterbi_path_trace(path_sel)
        return state, r, Nji

# %%
# s = humidity_latent_event_detection(x)

x_bg = np.ones(len(x)) * 50
obs_noise_var = 10
vi_model = hum_VI(x, x_bg, obs_noise_var)

state, r, N = vi_model.process()

print(f'k1 mu={vi_model.mu_k1}, lambda={vi_model.la_k1}')
print(f'k2 mu={vi_model.mu_k2}, lambda={vi_model.la_k2}')

fig, ax = plt.subplots(1)
ax.plot(x, label='humidity')
ax.set_xlabel('time step'); ax.set_ylabel('humidity [%]')
ax.set_title('humidity events')
ax2 = ax.twinx()
ax2.plot(true_s, color='C1', label='state')
ax2.plot(r[1,:]+0.01, color='C2', label='p(y=1) (BCJR)', alpha=0.6)
ax2.plot(state+0.02, color='C3', label='state (viterbi)', alpha=0.6)
ax2.set_ylabel('state')
ax2.legend()
plt.show()

