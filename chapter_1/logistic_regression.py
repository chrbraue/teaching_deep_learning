"""Logistische Regression: Logistische Regression auf Originalmerkmalen.

Aufruf aus der Repositorywurzel: python chapter_1/logistic_regression.py
Optional anschließend Regler öffnen: python chapter_1/logistic_regression.py --interactive
"""
import argparse
import matplotlib.pyplot as plt
from common import (exam_data, radial_data, fit_logistic, accuracy, configure_style,
                    draw_panel, finish, save_results, interactive, new_figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interactive', action='store_true')
    args = parser.parse_args()
    configure_style()
    x_left, y_left = exam_data()
    x, y = radial_data()
    w_left, b_left = fit_logistic(x_left, y_left)
    w, b = fit_logistic(x, y)
    fig, axes = new_figure()
    draw_panel(axes[0], x_left, y_left, w_left, b_left, (0, 105),
               ('Hausaufgabenpunkte in %', 'Anwesenheit in %'), 'Linear trennbare Daten')
    image = draw_panel(axes[1], x, y, w, b, (-15, 15), ('$x_1$', '$x_2$'),
                       'Nicht linear trennbare Daten')
    for ax, acc in zip(axes, [accuracy(x_left, y_left, w_left, b_left), accuracy(x, y, w, b)]):
        ax.text(.04, .96, f'Trainingsgenauigkeit: {acc:.0%}', transform=ax.transAxes,
                va='top', fontsize=6.5, bbox=dict(facecolor='white', edgecolor='none', alpha=.9))
    finish(fig, axes, image, 'logistic_regression', r'$\sigma(\boldsymbol{w}^\top\boldsymbol{x}+b)$', exam=True)
    save_results('logistic_regression_left', x_left, y_left, w_left, b_left, x_left, 12000)
    save_results('logistic_regression_right', x, y, w, b, x, 12000)
    plt.close(fig)
    if args.interactive:
        # Referenzen erhalten, damit die Regler während plt.show() aktiv bleiben.
        playground = interactive(x, y, w, b)
        plt.show()


if __name__ == '__main__':
    main()
