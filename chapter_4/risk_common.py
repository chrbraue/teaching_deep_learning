"""Gemeinsame schematische Kurven und Druckeinstellungen für Kapitel 4.

Keine Simulation oder empirischen Messdaten. Import erzeugt keine Dateien.
"""
from pathlib import Path
import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parent / 'output'
GENERALIZATION_COLOR = 'tab:blue'
ORANGE = 'tab:orange'


def risk_components(capacity):
    """Bayes-Risiko, Approximations- und Schätzfehler wie im bisherigen Beispiel."""
    capacity = np.asarray(capacity, dtype=float)
    bayes = np.full_like(capacity, 0.12)
    approximation = 0.65 * np.exp(-4 * capacity)
    estimation = 0.02 + 0.45 * capacity**2
    return bayes, approximation, estimation


def training_risk(capacity):
    """Ein zusätzlicher monoton fallender, rein schematischer Trainingsverlauf."""
    return 0.65 * np.exp(-5 * np.asarray(capacity, dtype=float)) + 0.02


def schematic_curves():
    """Beide Abbildungen verwenden exakt diese Arrays auf demselben Raster."""
    capacity = np.linspace(0, 1, 1001)
    bayes, approximation, estimation = risk_components(capacity)
    minimum_risk = bayes + approximation
    generalization = minimum_risk + estimation
    training = training_risk(capacity)
    return dict(capacity=capacity, bayes=bayes, approximation=approximation,
                estimation=estimation, minimum_risk=minimum_risk,
                generalization=generalization, training=training)


def setup_figure(curves):
    """Identische Achsen, Typografie und optimale Kapazität in beiden Grafiken."""
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.size': 9,
        'pgf.texsystem': 'pdflatex', 'pgf.rcfonts': False,
        'pgf.preamble': (r'\usepackage{amsmath,bm}'
                         r'\pdfinfoomitdate=1\pdftrailerid{}\pdfsuppressptexinfo=15'),
        'axes.spines.top': False, 'axes.spines.right': False,
    })
    fig, ax = plt.subplots(figsize=(6.3, 3.65))
    fig.subplots_adjust(left=.10, right=.98, bottom=.28, top=.92)
    capacity = curves['capacity']
    optimum = int(np.argmin(curves['generalization']))
    ax.axvline(capacity[optimum], color='black', lw=1, ls=(0, (5, 4)), ymax=.92)
    ax.text(capacity[optimum], -.045, 'opt. Kap.', ha='center', va='top',
            transform=ax.get_xaxis_transform())
    ax.text(.12, .92, 'Underfitting', transform=ax.transAxes, ha='left')
    ax.text(.95, .92, 'Overfitting', transform=ax.transAxes, ha='right')
    ax.text(0, 1.025, 'Schematische Verläufe', transform=ax.transAxes, fontsize=8)
    ax.set(xlabel='Kapazität', ylabel='Fehler', xlim=(0, 1), ylim=(0, .85),
           xticks=[], yticks=[])
    ax.xaxis.labelpad = 26
    return fig, ax


def plot_generalization(ax, curves):
    ax.plot(curves['capacity'], curves['generalization'],
            color=GENERALIZATION_COLOR, lw=2, ls='-',
            label=r'$\mathcal J_{\mathcal D}(f_{\mathcal S})$ (Generalisierungsfehler)')
    optimum = int(np.argmin(curves['generalization']))
    ax.plot(curves['capacity'][optimum], curves['generalization'][optimum],
            'o', color=GENERALIZATION_COLOR, ms=3)


def save_figure(fig, ax, stem, curves, annotation):
    fig.legend(*ax.get_legend_handles_labels(), loc='lower center', ncol=2,
               fontsize=7.2, frameon=False, columnspacing=1.0,
               labelspacing=.3, handletextpad=.5, borderpad=.2,
               bbox_to_anchor=(.5, .008))
    OUTPUT.mkdir(exist_ok=True)
    target = OUTPUT / (stem + '.pdf')
    fig.savefig(target, backend='pgf', metadata={'CreationDate': None, 'ModDate': None})
    subprocess.run(['pdftoppm', '-singlefile', '-r', '200', '-png',
                    str(target), str(target.with_suffix(''))], check=True)
    data = {name: values.tolist() for name, values in curves.items()}
    data.update(schematic_not_measured=True, annotations=annotation)
    target.with_suffix('.json').write_text(json.dumps(data, indent=2) + '\n')
    plt.close(fig)
