"""Schematische Risikozerlegung, keine empirischen Messergebnisse.

Aufruf: python chapter_4/risk_decomposition.py
Druckexport: NumPy, Matplotlib, pdflatex und pdftoppm (Poppler).
"""
from pathlib import Path
import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parent / 'output'


def risk_components(capacity):
    """Ein einfaches Schema, kein allgemeines Gesetz über reale Lernverfahren.

    Grau: bestes Risiko in der Hypothesenmenge = Bayes + Approximation.
    Grün: gelerntes Risiko = Bayes + Approximation + Schätzfehler.
    """
    capacity = np.asarray(capacity, dtype=float)
    bayes = np.full_like(capacity, 0.12)
    approximation = 0.65 * np.exp(-4 * capacity)
    estimation = 0.02 + 0.45 * capacity**2
    return bayes, approximation, estimation


def main():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.size': 9,
        'pgf.texsystem': 'pdflatex', 'pgf.rcfonts': False,
        'pgf.preamble': (r'\usepackage{amsmath,bm}'
                         r'\pdfinfoomitdate=1\pdftrailerid{}\pdfsuppressptexinfo=15'),
        'axes.spines.top': False, 'axes.spines.right': False,
    })
    capacity = np.linspace(0, 1, 1001)
    bayes, approximation, estimation = risk_components(capacity)
    best_in_class = bayes + approximation
    total = best_in_class + estimation
    optimum = int(np.argmin(total))

    fig, ax = plt.subplots(figsize=(6.3, 3.65))
    fig.subplots_adjust(left=.10, right=.98, bottom=.28, top=.92)
    ax.plot(capacity, best_in_class, color='#777777', lw=1.8, ls='--',
            label=r'$\mathcal J_{\mathcal D}(f_{\mathcal D})$ (bestes Klassenrisiko)')
    ax.plot(capacity, estimation, color='#c52b32', lw=1.8, ls='-.',
            label=r'$\varepsilon_{\mathrm{est}}$ (Schätzfehler)')
    ax.plot(capacity, total, color='#148536', lw=2,
            label=r'$\mathcal J_{\mathcal D}(f_{\mathcal S})$ (Gesamtrisiko)')
    ax.axhline(bayes[0], color='#333333', lw=1, ls=':',
               label=r'$\mathcal J_{\mathcal D}(f_{\mathrm B})$ (Bayes-Risiko)')
    ax.axvline(capacity[optimum], color='black', lw=1, ls=(0, (5, 4)), ymax=.92)
    ax.plot(capacity[optimum], total[optimum], 'o', color='#148536', ms=3)
    ax.text(capacity[optimum], -.045, 'opt. Kap.', ha='center', va='top',
            transform=ax.get_xaxis_transform())
    # Der Approximationsfehler ist die Differenz zur Bayes-Basis, nicht Grau selbst.
    position = .27
    base, app, _ = risk_components(position)
    ax.annotate('', xy=(position, float(base + app)), xytext=(position, float(base)),
                arrowprops=dict(arrowstyle='<->', color='#555555', lw=1))
    ax.text(position-.025, float(base + app/2), r'$\varepsilon_{\mathrm{app}}$',
            ha='right', va='center')
    ax.text(.12, .92, 'Underfitting', transform=ax.transAxes, ha='left')
    ax.text(.95, .92, 'Overfitting', transform=ax.transAxes, ha='right')
    ax.text(0, 1.025, 'Schematische Verläufe', transform=ax.transAxes, fontsize=8)
    ax.set(xlabel='Kapazität', ylabel='Fehler', xlim=(0, 1), ylim=(0, .85),
           xticks=[], yticks=[])
    ax.xaxis.labelpad = 26
    fig.legend(*ax.get_legend_handles_labels(), loc='lower center', ncol=2,
               fontsize=7.2, frameon=False, columnspacing=1.0,
               labelspacing=.3, handletextpad=.5, borderpad=.2,
               bbox_to_anchor=(.5, .008))
    OUTPUT.mkdir(exist_ok=True)
    target = OUTPUT / 'risk_decomposition.pdf'
    fig.savefig(target, backend='pgf', metadata={'CreationDate': None, 'ModDate': None})
    subprocess.run(['pdftoppm', '-singlefile', '-r', '200', '-png',
                    str(target), str(target.with_suffix(''))], check=True)
    (OUTPUT / 'risk_decomposition.json').write_text(json.dumps({
        'schematic_not_measured': True, 'bayes_risk': 0.12,
        'approximation': '0.65 * exp(-4 * capacity)',
        'estimation': '0.02 + 0.45 * capacity**2',
        'capacity_grid_points': len(capacity),
        'schematic_optimum_capacity': float(capacity[optimum]),
    }, indent=2) + '\n')
    plt.close(fig)


if __name__ == '__main__':
    main()
