"""Druckfassung von Soft-Thresholding und optionale Schieberegler.

python chapter_4/l1_regularization.py
python chapter_4/l1_regularization.py --interactive
"""
from pathlib import Path
import argparse
import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

OUTPUT = Path(__file__).resolve().parent / 'output'
# Normierte Druckwerte. Die Vorlage enthält nur symbolische Achsenangaben.
DEFAULT_GAMMA = 1.0
DEFAULT_WEIGHT = 1.0
DEFAULT_CURVATURE = 1.0


def soft_threshold(w_hat, gamma, h):
    """Minimiert h/2 * (w-w_hat)**2 + gamma*abs(w) für h=H_ii>0."""
    w_hat = np.asarray(w_hat, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    h = float(h)
    if not np.isfinite(h) or h <= 0:
        raise ValueError('H_ii must be finite and positive.')
    if not np.all(np.isfinite(gamma)) or np.any(gamma < 0):
        raise ValueError('gamma must be finite and nonnegative.')
    if not np.all(np.isfinite(w_hat)):
        raise ValueError('w_hat must be finite.')
    return np.sign(w_hat) * np.maximum(np.abs(w_hat) - gamma/h, 0.0)


def configure_typography():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.size': 9,
        'mathtext.fontset': 'cm',
        'pgf.texsystem': 'pdflatex', 'pgf.rcfonts': False,
        'pgf.preamble': (r'\usepackage{amsmath,bm}'
                         r'\pdfinfoomitdate=1\pdftrailerid{}\pdfsuppressptexinfo=15'),
    })


def damping_arrow(ax, x, old, new):
    if not np.isclose(old, new):
        ax.annotate('', xy=(x, new), xytext=(x, old),
                    arrowprops=dict(arrowstyle='-|>', color='tab:blue', lw=.65,
                                    mutation_scale=7, shrinkA=0, shrinkB=0))


def draw_panels(axes, gamma, w_hat, h, *, interactive=False):
    """Beide Originalansichten, dieselbe Funktion auch bei Slider-Änderungen."""
    left, right = axes
    for ax in axes:
        ax.clear()
        ax.tick_params(direction='in', top=True, right=True, labelsize=9)
    threshold = gamma/h
    extent = max(2.0, 2*threshold, 1.2*abs(w_hat)) if interactive else 2.0
    weights = np.linspace(-extent, extent, 501)
    left.plot(weights, weights, color='#333333', lw=1.1, ls=(0, (5, 4)))
    left.plot(weights, soft_threshold(weights, gamma, h), color='tab:orange',
              lw=2.2, gid='regularized')
    for x in np.linspace(-extent, extent, 17):
        damping_arrow(left, x, x, float(soft_threshold(x, gamma, h)))
    left.set(xlim=(-extent, extent), ylim=(-extent, extent),
             xlabel=r'$\hat w_i$',
             ylabel=r'$(w_\gamma)_i=\mathrm{prox}_{\frac{\gamma}{H_{ii}}|\cdot|}(\hat w_i)$',
             yticks=[0], yticklabels=['0'])
    if threshold > 0:
        left.set_xticks([-threshold, threshold],
                        [r'$-\frac{\gamma}{H_{ii}}$', r'$\frac{\gamma}{H_{ii}}$'])
    else:
        left.set_xticks([0], ['0'])

    gamma_max = max(2.0, 1.25*h*abs(w_hat), 1.1*gamma) if interactive else 2.0
    gammas = np.linspace(0, gamma_max, 501)
    right.axhline(w_hat, color='#333333', lw=1.1, ls=(0, (5, 4)))
    right.plot(gammas, soft_threshold(w_hat, gammas, h), color='tab:orange',
               lw=2.2, gid='regularized')
    for x in np.linspace(.05*gamma_max, .95*gamma_max, 13):
        damping_arrow(right, x, w_hat, float(soft_threshold(w_hat, x, h)))
    if w_hat > 0:
        right.set_ylim(0, 1.05*w_hat)
    elif w_hat < 0:
        right.set_ylim(1.05*w_hat, 0)
    else:
        right.set_ylim(-1, 1)
    right.set(xlim=(0, gamma_max), xlabel=r'$\gamma$')
    if w_hat:
        right.set_yticks([0, w_hat], ['0', r'$\hat w_i$'])
    else:
        right.set_yticks([0], ['0'])
    if w_hat:
        label = r'$H_{ii}|\hat w_i|$' if interactive else r'$H_{ii}\hat w_i$'
        right.set_xticks([h*abs(w_hat)], [label])
    else:
        right.set_xticks([0], ['0'])
    if interactive:
        value = float(soft_threshold(w_hat, gamma, h))
        left.plot(w_hat, value, 'o', color='tab:blue', ms=5, gid='current')
        right.plot(gamma, value, 'o', color='tab:blue', ms=5, gid='current')


def export_static():
    configure_typography()
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.35))
    fig.subplots_adjust(left=.075, right=.985, bottom=.23, top=.97, wspace=.20)
    draw_panels(axes, DEFAULT_GAMMA, DEFAULT_WEIGHT, DEFAULT_CURVATURE)
    OUTPUT.mkdir(exist_ok=True)
    target = OUTPUT / 'l1_regularization.pdf'
    fig.savefig(target, backend='pgf', metadata={'CreationDate': None, 'ModDate': None})
    subprocess.run(['pdftoppm', '-singlefile', '-r', '200', '-png',
                    str(target), str(target.with_suffix(''))], check=True)
    target.with_suffix('.json').write_text(json.dumps({
        'gamma_left': DEFAULT_GAMMA, 'w_hat_right': DEFAULT_WEIGHT,
        'H_ii': DEFAULT_CURVATURE, 'normalized_values_not_recovered_parameters': True,
        'left_weight_range': [-2, 2], 'right_gamma_range': [0, 2],
    }, indent=2) + '\n')
    plt.close(fig)


def create_interactive():
    configure_typography()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.6))
    fig.canvas.manager.set_window_title('Soft-Thresholding – ℓ₁-Regularisierung')
    fig.subplots_adjust(left=.10, right=.97, bottom=.40, top=.94, wspace=.28)
    sliders = {
        'gamma': Slider(fig.add_axes([.17, .25, .65, .035]), r'$\gamma$', 0, 3,
                        valinit=DEFAULT_GAMMA),
        'w_hat': Slider(fig.add_axes([.17, .18, .65, .035]), r'$\hat w_i$', -2, 2,
                        valinit=DEFAULT_WEIGHT),
        'h': Slider(fig.add_axes([.17, .11, .65, .035]), r'$H_{ii}$', .25, 4,
                    valinit=DEFAULT_CURVATURE),
    }
    readout = fig.text(.10, .035, '', fontsize=10)
    reset = Button(fig.add_axes([.84, .025, .13, .06]), 'Reset')

    def update(_=None):
        gamma, w_hat, h = (sliders[k].val for k in ['gamma', 'w_hat', 'h'])
        draw_panels(axes, gamma, w_hat, h, interactive=True)
        value = float(soft_threshold(w_hat, gamma, h))
        value = 0.0 if value == 0 else value
        readout.set_text(r'$\gamma/H_{ii}$ = ' + f'{gamma/h:.3f}'
                         + r'     $(w_\gamma)_i$ = ' + f'{value:.3f}')
        fig.canvas.draw_idle()

    def reset_values(_):
        for slider in sliders.values():
            slider.reset()

    for slider in sliders.values():
        slider.on_changed(update)
    reset.on_clicked(reset_values)
    update()
    # Widgets müssen während plt.show() referenziert bleiben.
    fig._widgets = (sliders, reset)
    return fig, sliders, reset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interactive', action='store_true', help='Show optional sliders.')
    args = parser.parse_args()
    export_static()
    if args.interactive:
        create_interactive()
        plt.show()


if __name__ == '__main__':
    main()
