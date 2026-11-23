"""Quadratische Basisfunktionen: Eine Hypothese im Merkmals- und im ursprünglichen Objektraum.

Aufruf: python chapter_1/logistic_regression_basis_functions.py [--interactive]
"""
import argparse
import matplotlib.pyplot as plt
from common import (radial_data, phi, fit_logistic, accuracy, configure_style,
                    draw_panel, finish, save_results, interactive, new_figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interactive', action='store_true')
    args = parser.parse_args()
    configure_style()
    x, y = radial_data()  # Exakt dieselbe Trainingsmenge wie im Beispiel auf Originalmerkmalen.
    features = phi(x)
    w, b = fit_logistic(features, y, steps=100000)
    fig, axes = new_figure(probability_labels=True)
    feature_limit = float(features.max()) * 1.06
    draw_panel(axes[0], features, y, w, b, (-6, feature_limit),
               (r'$\phi_1$', r'$\phi_2$'), 'Quadratischer Merkmalsraum')
    image = draw_panel(axes[1], x, y, w, b, (-15, 15), ('$x_1$', '$x_2$'),
                       'Ursprünglicher Objektraum', transform=phi)
    # Derselbe Score, einmal in phi-Koordinaten, einmal als Funktion von x.
    axes[0].set_xlabel(r'$\phi_1$' + '\n' +
                       r'$\sigma(\boldsymbol{w}^\top\boldsymbol{\phi}+b)$', labelpad=2)
    axes[1].set_xlabel(r'$x_1$' + '\n' +
                       r'$\sigma(\boldsymbol{w}^\top\boldsymbol{\phi}(\boldsymbol{x})+b)$', labelpad=2)
    acc = accuracy(features, y, w, b)
    fig.suptitle(f'Dieselbe Hypothese in beiden Darstellungen · Trainingsgenauigkeit: {acc:.0%}', fontsize=7.5, y=0.995)
    finish(fig, axes, image, 'logistic_regression_basis_functions', 'Modellwahrscheinlichkeit')
    save_results('logistic_regression_basis_functions', x, y, w, b, features, 100000)
    plt.close(fig)
    if args.interactive:
        playground = interactive(x, y, w, b, quadratic=True)
        plt.show()


if __name__ == '__main__':
    main()
