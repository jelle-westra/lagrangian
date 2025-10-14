from abc import ABC, abstractmethod
from tqdm import trange

import torch
from torch.autograd.functional import jacobian, hessian


class LagrangianSolver(ABC):
    @abstractmethod
    def T(self, u: torch.Tensor) -> torch.Tensor: 
        pass

    @abstractmethod
    def V(self, u: torch.Tensor) -> torch.Tensor: 
        pass

    def Q(self, u: torch.Tensor) -> torch.Tensor: 
        # overwrite this for energy dissipation
        return torch.tensor(0.)

    def energy(self, t, q, qdot) -> torch.Tensor:
        u = torch.tensor([t, *q, *qdot], requires_grad=True, dtype=torch.float64)
        with torch.no_grad(): 
            return self.T(u) + self.V(u)

    def lagrangian(self, u: torch.Tensor) -> torch.Tensor:
        return self.T(u) - self.V(u)
    
    def qddot(self, t, q, qdot):
        u = torch.tensor([t, *q, *qdot], requires_grad=True, dtype=torch.float64)
        H = hessian(self.lagrangian, u)
        
        # J_L = dL/dq
        J_L = jacobian(self.lagrangian, u)[1:self.n+1]
        
        # J_Q = dQ/dq_dot (dissipative forces)
        J_Q = jacobian(self.Q, u)[self.n+1:]
        
        # Generalized forces F = J_L - J_Q - H @ qdot
        F = (J_L - J_Q) - H[self.n+1:, 1:self.n+1] @ qdot

        M_inv = torch.inverse(H[self.n+1:, self.n+1:])
        qddot = (M_inv @ F)
        return qddot
    
    def solve(self, u0: torch.Tensor, t: torch.Tensor) -> tuple:
        self.n = u0.size()[0]//2

        U = torch.empty((len(t), *u0.size()))
        U[0] = u0

        (q, qdot) = u0.view(2, -1)
        qddot = self.qddot(t[0], q, qdot)

        E = torch.empty((len(t)))
        E0 = self.energy(t[0], q, qdot)
        E[0] = E0

        t1 = t[0]
        for i in trange(1, len(t)):
            t2 = t[i]
            dt = (t2 - t1)

            # implicit midpoint iteration
            q_mid = q.clone()
            qdot_mid = qdot.clone()

            for _ in range(5):  # usually converges very fast
                q_mid_prev, qdot_mid_prev = q_mid.clone(), qdot_mid.clone()

                # evaluate acceleration at midpoint
                qddot_mid = self.qddot(t2 - 0.5 * dt, q_mid, qdot_mid)

                # midpoint updates (Newton iteration)
                q_mid = q + 0.5 * dt * qdot_mid
                qdot_mid = qdot + 0.5 * dt * qddot_mid

                # convergence check
                if torch.norm(q_mid - q_mid_prev) < 1e-10 and torch.norm(qdot_mid - qdot_mid_prev) < 1e-10:
                    break

            # final full step using midpoint values
            q = q + dt * qdot_mid
            qdot = qdot + dt * self.qddot(t2 - 0.5 * dt, q_mid, qdot_mid)

            U[i,:self.n] = q
            U[i,self.n:] = qdot
            E[i] = self.energy(t2, q, qdot)
            t1 = t2

        return (t, U, E)