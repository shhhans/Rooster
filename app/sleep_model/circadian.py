"""Forger99 equations. Source snapshot and independent reference in sources/."""
import numpy as np
from scipy.integrate import solve_ivp

PARAMETERS = {'taux': 24.2, 'mu': .23, 'G': 33.75, 'alpha_0': .05,
              'beta': .0075, 'p': .5, 'I0': 9500., 'k': .55, 'cbt_to_dlmo': 7.}


def rhs(t, state, light):
    if light < 0 or not np.isfinite(light):
        raise ValueError('invalid photopic light')
    x, xc, n = state
    p = PARAMETERS
    alpha = p['alpha_0'] * (light / p['I0']) ** p['p']
    B = p['G'] * (1-n)*alpha*(1-.4*x)*(1-.4*xc)
    return np.array([np.pi/12*(xc+B),
                     np.pi/12*(p['mu']*(xc-4/3*xc**3)-x*((24/(.99669*p['taux']))**2+p['k']*B)),
                     60*(alpha*(1-n)-p['beta']*n)])


def integrate(state, hours, light, config, samples=None):
    sol = solve_ivp(lambda t, y: rhs(t, y, light), (0, hours), state,
                    method='DOP853', rtol=config['rtol'], atol=config['atol'],
                    max_step=config['max_step_minutes']/60, dense_output=True)
    if not sol.success or not np.all(np.isfinite(sol.y)):
        raise ArithmeticError('circadian integration failed: ' + sol.message)
    return sol.y[:, -1], sol.sol(samples) if samples is not None else None
