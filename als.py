import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Callable

# Convergence Tracker
@dataclass
class ALSLog:
  solver_name   : str
  train_rmse    : List[float] = field(default_factory=list)
  test_rmse     : List[float] = field(default_factory=list)
  train_loss    : List[float] = field(default_factory=list)
  residual_norm : List[float] = field(default_factory=list)
  cond_number   : List[float] = field(default_factory=list)
  elapsed_time  : List[float] = field(default_factory=list)


  def rmse_delta(self):
    tr = self.train_rmse
    return [abs(tr[i] - tr[i-1]) for i in range(1, len(tr))]


  def epoch_to_convergence(self, eps=1e-4):
    for i, d in enumerate(self.rmse_delta()):
      if d < eps: return i + 2
    return None


# Metric Helpers
def _rmse(R_true, U, V):
  mask = R_true > 0
  pred = (U @ V.T)[mask]
  return float(np.sqrt(np.mean((pred - R_true[mask]) ** 2)))


def _reg_loss(R, U, V, lam):
  mask = R > 0
  pred = (U @ V.T)[mask]
  return float(np.sum((pred - R[mask]) ** 2) 
               + lam * (np.sum(U ** 2) + np.sum(V ** 2)))


def _safe_cond(A):
  try:
    return float(np.linalg.cond(A))
  except np.linalg.LinAlgError:
    return float('inf')


# ALS Engine
def run_als(R_train, R_test, solver_fn, solver_name, k=10, lam=0.1, n_epochs=50, seed=42, verbose=True):
  log = ALSLog(solver_name=solver_name)
  np.random.seed(seed)
  n_users, n_items = R_train.shape
  lam_I = lam * np.eye(k)

  # Initialize latent factors
  U = np.random.normal(0, 0.1, (n_users, k))
  V = np.random.normal(0, 0.1, (n_items, k))

  # Pre-compute rated indices
  user_items = [np.where(R_train[u] > 0)[0] for u in range(n_users)]
  item_users = [np.where(R_train[:, i] > 0)[0] for i in range(n_items)]

  t0 = time.time()

  for epoch in range(n_epochs):
    ep_res, ep_cond = [], []

    # Fix V, update all user rows
    for u in range(n_users):
      rated = user_items[u]
      if len(rated) == 0: continue

      V_Iu = V[rated]
      r_u  = R_train[u, rated]
      A    = V_Iu.T @ V_Iu + lam_I
      b    = V_Iu.T @ r_u

      x, res = solver_fn(A, b)
      U[u] = x
      ep_res.append(res)
      ep_cond.append(_safe_cond(A))

    # Fix U, update all item rows
    for i in range(n_items):
      rated = item_users[i]
      if len(rated) == 0: continue

      U_Ii = U[rated]
      r_i  = R_train[rated, i]
      A    = U_Ii.T @ U_Ii + lam_I
      b    = U_Ii.T @ r_i

      x, res = solver_fn(A, b)
      V[i] = x
      ep_res.append(res)
      ep_cond.append(_safe_cond(A))

    # Log metrics
    log.train_rmse.append(_rmse(R_train, U, V))
    log.test_rmse.append(_rmse(R_test, U, V))
    log.train_loss.append(_reg_loss(R_train, U, V, lam))
    log.residual_norm.append(float(np.mean(ep_res)))
    log.cond_number.append(float(np.mean(ep_cond)))
    log.elapsed_time.append(time.time() - t0)

    if verbose and (epoch+1) % 5 == 0:
      print(f'Epoch {epoch+1:3d} | '
            f'Train RMSE : {log.train_rmse[-1]:.4f} | '
            f'Test RMSE  : {log.test_rmse[-1]:.4f} | '
            f'Loss       : {log.train_loss[-1]:.2f} | '
            f'kappa(A)   : {log.cond_number[-1]:.2f}')
      
  return log