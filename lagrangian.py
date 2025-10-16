import torch
from torch.autograd.functional import jacobian, hessian

from typing import Tuple
from tqdm import trange

class LagrangianSolver():
    n: int
    n_constraints: int

    def T(self, u: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0., dtype=torch.float64)
    
    def V(self, u: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0., dtype=torch.float64)
    
    def Q(self, u: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0., dtype=torch.float64)

    def constraints(self, u: torch.Tensor) -> torch.Tensor :
        raise NotImplementedError()

    def lagrangian(self, u: torch.Tensor) -> torch.Tensor: 
        return self.T(u) - self.V(u)
    
    def energy(self, u: torch.Tensor) -> torch.Tensor: 
        return self.T(u) + self.V(u)
    
    def qddot(self, 
        t: torch.Tensor, 
        q: torch.Tensor, 
        qdot: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor] :
        # returns (qddot, lambda)
        # langrange multipliers are solved numerically alongside qddot if system is constrained

        u = torch.tensor([t, *q, *qdot], requires_grad=True, dtype=torch.float64)

        H = hessian(self.lagrangian, u)
        M = H[self.n+1:, self.n+1:]
        
        # J_L = dL/dq
        J_L = jacobian(self.lagrangian, u)[1:self.n+1]
        
        # J_Q = dQ/dq_dot (dissipative forces)
        J_Q = jacobian(self.Q, u)[self.n+1:]

        # Generalized forces F = J_L - J_Q - H @ qdot
        F = (J_L - J_Q) - H[self.n+1:, 1:self.n+1] @ qdot

        if (self.n_constraints):
            Jg = jacobian(self.constraints, u)[:,1:self.n+1]

            # TODO : this is not the most efficient way of doing ways
            # g is computed `n_constraints` times now
            gamma = torch.zeros(self.n_constraints, dtype=torch.float64)
            for i in range(self.n_constraints):
                g_i = lambda u_in: self.constraints(u_in)[i]
                H_gi = hessian(g_i, u)[1:self.n+1, 1:self.n+1]
                gamma[i] = -qdot @ H_gi @ qdot

            #     Ax = b
            #  [ M    -Jg^T ] [ qddot ]   [  F   ]
            #  [ Jg     0   ] [ lambda] = [ gamma]

            A = torch.zeros((self.n + self.n_constraints, self.n + self.n_constraints), dtype=torch.float64)
            A[:self.n, :self.n] = M
            A[:self.n, self.n:] = -Jg.T
            A[self.n:, :self.n] = Jg

            b = torch.cat([F, gamma])
            x = torch.linalg.solve(A, b)

            (qddot, lambdas) = (x[:self.n], x[self.n:])
            return (qddot, lambdas)
        else:
            qddot = torch.linalg.solve(M, F)
            return (qddot, None)
    
    def solve(self, u0: torch.Tensor, t: torch.Tensor) -> tuple:
        self.n = u0.size()[0]//2
        try:
            self.n_constraints = len(self.constraints(torch.tensor([t[0], *u0], dtype=torch.float64)))
            assert self.n_constraints >= 1, 'constraints must have dim >= 1 for vector operations, use unsqueeze to add dimension'
        except NotImplementedError:
            self.n_constraints = 0

        self.log = {'t': t}
        self.log['u'] = torch.empty((len(t), *u0.size()))
        self.log['u'][0] = u0


        self.log['E'] = torch.empty((len(t)))
        self.log['E'][0] = self.energy(torch.tensor([t[0], *u0], dtype=torch.float64))

        if (self.n_constraints):
            self.log['c'] = torch.zeros((len(t), self.n_constraints))
            self.log['lambdas'] = torch.zeros((len(t), self.n_constraints), dtype=float)

        t1 = t[0]
        (q, qdot) = u0.view(2, self.n)
        for i in trange(1, len(t)):
            t2 = t[i]
            dt = (t2 - t1)

            # implicit midpoint iteration
            q_mid = q.clone()
            qdot_mid = qdot.clone()

            for _ in range(5):  # usually converges very fast
                q_mid_prev, qdot_mid_prev = q_mid.clone(), qdot_mid.clone()

                # evaluate acceleration at midpoint
                (qddot_mid, _) = self.qddot(t2 - 0.5 * dt, q_mid, qdot_mid)

                # midpoint updates (Newton iteration)
                q_mid = q + 0.5 * dt * qdot_mid
                qdot_mid = qdot + 0.5 * dt * qddot_mid

                # convergence check
                if torch.norm(q_mid - q_mid_prev) < 1e-10 and torch.norm(qdot_mid - qdot_mid_prev) < 1e-10:
                    break

            # final full step using midpoint values
            q = q + dt * qdot_mid

            (qddot, lambdas) = self.qddot(t2 - 0.5 * dt, q_mid, qdot_mid)
            qdot = qdot + dt * qddot

            self.log['u'][i,:self.n] = q
            self.log['u'][i,self.n:] = qdot
            self.log['E'][i] = self.energy(torch.tensor([t2, *self.log['u'][i]]))
            if (self.n_constraints):
                self.log['c'][i] = self.constraints(torch.tensor([t2, *self.log['u'][i]]))
                self.log['lambdas'][i] = lambdas

            t1 = t2

        return self.log