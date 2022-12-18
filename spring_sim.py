from typing import List, Union

import numpy as np
import scipy as sp
from autograd import grad
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from spring_drawing import spring


def phi(r: float) -> float:
    if r < 0:
        return k1 * r ** 2 / 2
    return k2 * r ** 2 / 2


def psi(v: np.ndarray) -> float:
    return phi(v[len(nodes_1) - 1] - v[len(nodes_1) - 2])


def get_grid(nodes_density, a, b, l):
    nodes_1 = np.arange(a, -l + 1e-10, 1 / nodes_density)
    nodes_2 = np.arange(l, b + 1e-10, 1 / nodes_density)
    return nodes_1, nodes_2


def solve(nodes_1, nodes_2, l, E1, E2, f1_coeff, f2_coeff):
    def f1(x: np.ndarray) -> np.ndarray:
        return np.full_like(x, f1_coeff)

    def f2(x: np.ndarray) -> np.ndarray:
        return np.full_like(x, f2_coeff)

    f1_h = f1(nodes_1)
    f2_h = f2(nodes_2)
    A = np.zeros((len(nodes_1) + len(nodes_2) - 2,) * 2)
    F = np.zeros((len(nodes_1) + len(nodes_2) - 2,))
    for i in range(len(nodes_1) - 2):
        A[i, i] = E1 * 2 * nodes_density
        A[i, i + 1] = A[i + 1, i] = -E1 * nodes_density
    A[len(nodes_1) - 2, len(nodes_1) - 2] = E1 * nodes_density

    F[:len(nodes_1) - 1] = f1_h[1:] / (2 * nodes_density)

    for i in range(len(nodes_1), len(nodes_1) + len(nodes_2) - 2):
        A[i, i] = E2 * 2 * nodes_density
        A[i, i - 1] = A[i - 1, i] = -E2 * nodes_density
    A[len(nodes_1) - 1, len(nodes_1) - 1] = E2 * nodes_density

    F[len(nodes_1) - 1:] = f2_h[:-1] / (2 * nodes_density)

    grad_psi = grad(psi)

    constraint_A = np.zeros_like(F)
    constraint_A[len(nodes_1) - 2] = 1
    constraint_A[len(nodes_1) - 1] = -1

    constraint = sp.optimize.LinearConstraint(constraint_A, ub=2 * l)
    sol = sp.optimize.minimize(lambda v: .5 * v @ A @ v - F @ v + psi(v), np.random.randn(*F.shape),
                               options={"disp": True, "maxiter": 10000}, jac=lambda v: A @ v - F + grad_psi(v),
                               constraints=[constraint])
    new_position = np.hstack([nodes_1, nodes_2])
    new_position[1:-1] += sol.x
    return new_position, sol.x


def draw(solutions: Union[np.ndarray, List], stresses: Union[np.ndarray, List], nodes_1, nodes_2, E1s, E2s, k1s, k2s,
         f1_coeffs, f2_coeffs):
    fig, axs = plt.subplots(len(solutions), 1, sharex=True, figsize=(7, 1 + 2 * len(solutions)))
    for i, (solution, stress) in enumerate(zip(solutions, stresses)):
        new_position = np.hstack([nodes_1, nodes_2])
        new_position[1:-1] += solution
        stress_1 = np.zeros_like(nodes_1)
        stress_2 = np.zeros_like(nodes_2)
        stress_1[1:] = stress[:len(nodes_1) - 1]
        stress_1[0] = stress_1[1]
        stress_2[:-1] = stress[len(nodes_1) - 1:]
        stress_2[-1] = stress_2[-2]

        axs[i].set_ylim([-1, 3])
        axs[i].set_aspect(1)
        axs[i].set_yticks([])
        points_1 = np.array([new_position[:len(nodes_1)], np.zeros(len(nodes_1)) + 0.01]).T.reshape([-1, 1, 2])
        segments_1 = np.concatenate([points_1[:-1], points_1[1:]], axis=1)
        norm = plt.Normalize(-30, 30)
        axs[i].scatter(new_position[:len(nodes_1)], np.zeros(len(nodes_1)), marker='.', c=stress_1, cmap="cool",
                       norm=norm)
        lc_1 = LineCollection(segments_1, cmap='cool', norm=norm)
        lc_1.set_array(stress_1)
        lc_1.set_linewidth(1)
        axs[i].add_collection(lc_1)

        points_2 = np.array([new_position[len(nodes_1):], np.zeros(len(nodes_2)) + 0.01]).T.reshape([-1, 1, 2])
        segments_2 = np.concatenate([points_2[:-1], points_2[1:]], axis=1)
        axs[i].scatter(new_position[len(nodes_1):], np.zeros(len(nodes_2)), marker='.', c=stress_2, cmap="viridis",
                       norm=norm)
        lc_2 = LineCollection(segments_2, cmap='viridis', norm=norm)
        lc_2.set_array(stress_2)
        lc_2.set_linewidth(1)
        axs[i].add_collection(lc_2)
        axs[i].plot(*spring([new_position[len(nodes_1) - 1], 0], [new_position[len(nodes_1)], 0], 16, 1), c="black")
        axs[i].text(a, 1.5,
                    f"a={a} b={b} l={l}\nf1={f1_coeffs[i]} f2={f2_coeffs[i]} k1={k1s[i]} k2={k2s[i]} E1={E1s[i]} E2={E2s[i]}",
                    fontfamily="monospace")
    fig.subplots_adjust(bottom=0.15)
    cb1_ax = fig.add_axes([0.1, 0.075, 0.8, 0.025])
    cb2_ax = fig.add_axes([0.1, 0.05, 0.8, 0.025])
    fig.colorbar(lc_1, cax=cb1_ax, orientation="horizontal", ticks=[])
    fig.colorbar(lc_2, cax=cb2_ax, orientation="horizontal")
    plt.savefig("no_sim_k.png")


if __name__ == '__main__':
    l = 4
    a = -8
    b = 10
    nodes_density = 4

    nodes_1, nodes_2 = get_grid(nodes_density, a, b, l)

    E1s = [20] * 5
    E2s = [10] * 5
    k1s = [1, 3, 8, 20, 1e6]
    k2s = [1] * 5

    f1_coeffs = [15] * 5
    f2_coeffs = [-6] * 5

    solutions = []
    stresses = []
    for E1, E2, k1, k2, f1_coeff, f2_coeff in zip(E1s, E2s, k1s, k2s, f1_coeffs, f2_coeffs):
        _, solution = solve(nodes_1, nodes_2, l, E1, E2, f1_coeff, f2_coeff)
        solutions.append(solution)
        solution_padded = np.append(np.insert(solution, 0, 0), 0)
        stress = np.zeros_like(solution)
        stress[:len(nodes_1) - 1] = E1 * (
                    solution_padded[1:len(nodes_1)] - solution_padded[:len(nodes_1) - 1]) * nodes_density
        stress[len(nodes_1) - 1:] = E2 * (
                    solution_padded[len(nodes_1) + 1:] - solution_padded[len(nodes_1):-1]) * nodes_density
        stresses.append(stress)
    draw(solutions, stresses, nodes_1, nodes_2, E1s, E2s, k1s, k2s, f1_coeffs, f2_coeffs)
