"""Numerische Qualitätskontrolle: python chapter_1/validate.py."""
from itertools import combinations
import json
import numpy as np
import matplotlib.pyplot as plt
from common import (exam_data, radial_data, phi, fit_logistic, accuracy,
                    sigmoid, interactive)


def nonseparability_certificate(x, y):
    """Ein innerer Punkt als echte Konvexkombination dreier äußerer Punkte.

    Wären alle äußeren Scores >= 0, wäre auch der Score ihrer Konvexkombination
    >= 0. Der innere Punkt benötigt aber Score < 0: ein Widerspruch.
    """
    inner = int(np.flatnonzero(y == 0)[0])
    for indices in combinations(np.flatnonzero(y == 1), 3):
        matrix = np.vstack([x[list(indices)].T, np.ones(3)])
        if abs(np.linalg.det(matrix)) < 1e-10:
            continue
        coefficients = np.linalg.solve(matrix, np.r_[x[inner], 1])
        if np.min(coefficients) > 1e-4:
            assert np.allclose(coefficients @ x[list(indices)], x[inner], atol=1e-10)
            return {'inner_index': inner, 'outer_indices': [int(i) for i in indices],
                    'coefficients': coefficients.tolist()}
    raise AssertionError('Kein Zertifikat gefunden')


def main():
    x, y = radial_data()
    x_again, y_again = radial_data()
    assert np.array_equal(x, x_again) and np.array_equal(y, y_again)
    certificate = nonseparability_certificate(x, y)
    features = phi(x)
    assert np.min((2*y - 1) * (features.sum(axis=1) - 36)) > 0
    w, b = fit_logistic(features, y, steps=100000)
    assert accuracy(features, y, w, b) == 1
    assert np.all(w > 0) and b < 0
    # Identische Scores im Merkmalsraum und bei Auswertung phi(x).
    assert np.allclose(features @ w + b, w[0]*x[:, 0]**2 + w[1]*x[:, 1]**2 + b)
    angles = np.linspace(0, 2*np.pi, 101)
    boundary = np.column_stack([np.sqrt(-b/w[0])*np.cos(angles),
                                np.sqrt(-b/w[1])*np.sin(angles)])
    assert np.allclose(sigmoid(phi(boundary) @ w+b), .5, atol=1e-12)
    x_left, y_left = exam_data()
    wl, bl = fit_logistic(x_left, y_left)
    assert accuracy(x_left, y_left, wl, bl) == 1
    wr, br = fit_logistic(x, y)
    assert accuracy(x, y, wr, br) < 1
    # Lineare Nulllinie parametrisieren (hier ist w1 von null verschieden).
    x2 = np.linspace(-15, 15, 20)
    line = np.column_stack([-(wr[1]*x2+br)/wr[0], x2])
    assert np.allclose(sigmoid(line @ wr+br), .5, atol=1e-12)
    # Keine GUI nötig: tatsächliche Regler und ihre Callbacks betätigen.
    for weights, bias, quadratic in [(wr, br, False), (w, b, True)]:
        fig, sliders, update = interactive(x, y, weights, bias, quadratic)
        sliders[0].set_val(weights[0] * .8)
        sliders[1].set_val(weights[1] * .8)
        sliders[2].set_val(bias * .8)
        update()
        assert len(fig.axes[0].collections) >= 3
        plt.close(fig)
    result = {'original_not_linearly_separable': certificate,
              'quadratic_separator': {'w': [1, 1], 'b': -36},
              'accuracy': {'logistic_regression_left': 1.0, 'logistic_regression_right': accuracy(x, y, wr, br),
                           'logistic_regression_basis_functions': accuracy(features, y, w, b)},
              'identical_data': True, 'same_hypothesis': True,
              'boundary_probability_verified': True, 'slider_callbacks_verified': True}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
