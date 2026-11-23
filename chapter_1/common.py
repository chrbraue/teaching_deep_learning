"""Gemeinsame Daten, Training und Darstellung für die Beispiele zur logistischen Regression."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

SEED = 7
RADIUS = 6.0
OUTPUT = Path(__file__).resolve().parent / 'output'


def exam_data():
    """Fest vorgegebenes synthetisches Beispiel, keine realen Prüfungsdaten."""
    failed = np.array([[1,11], [3,73], [9,49], [10,51], [12,25], [16,83],
                       [20,70], [20,27], [21,75], [30,78], [35,20],
                       [54,26], [59,33], [78,22], [96,1]], dtype=float)
    passed = np.array([[43,74], [57,90], [69,73], [72,85],
                       [84,55], [91,63], [93,98], [100,100]], dtype=float)
    return np.vstack([failed, passed]), np.r_[np.zeros(len(failed)), np.ones(len(passed))]


def radial_data():
    """Trainingsmenge S: 100 Objekte x und binäre Labels y, in beiden Figuren identisch.

    Klasse 0 liegt innerhalb des Radius, Klasse 1 außerhalb (Rand: Klasse 0).
    Der Generator wird bei jedem Aufruf neu mit demselben Seed initialisiert.
    """
    x = np.random.default_rng(SEED).normal(0, 5, size=(100, 2))
    y = (np.sum(x**2, axis=1) > RADIUS**2).astype(int)
    return x, y


def phi(x):
    """Basisfunktionen phi_1(x) = x_1² und phi_2(x) = x_2²."""
    return np.asarray(x)**2


def sigmoid(z):
    return np.exp(-np.logaddexp(0, -z))


def fit_logistic(x, y, steps=12000, learning_rate=0.3):
    """Mittlere binäre Kreuzentropie mit vollem Gradientenverfahren minimieren.

    Intern standardisieren; zurückgegebene theta=(w,b) gelten in den übergebenen
    Koordinaten. Keine Regularisierung, kein Download, kein Framework.
    Bei trennbaren Daten gibt es kein endliches unregularisiertes ML-Optimum:
    Wir zeigen einen festgelegten endlichen Trainingsstand.
    """
    mean, scale = x.mean(axis=0), x.std(axis=0)
    z = (x - mean) / scale
    w, b = np.zeros(2), 0.0
    for _ in range(steps):
        error = sigmoid(z @ w + b) - y
        w -= learning_rate * (z.T @ error) / len(y)
        b -= learning_rate * error.mean()
    return w / scale, float(b - mean @ (w / scale))


def predict(x, w, b):
    """h_theta: Klasse 1 für Score >= 0, sonst Klasse 0 (wie im Skript)."""
    return (x @ w + b >= 0).astype(int)


def accuracy(x, y, w, b):
    return float(np.mean(predict(x, w, b) == y))


def configure_style():
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                         'mathtext.fontset': 'cm', 'pgf.texsystem': 'pdflatex',
                         'pgf.rcfonts': False,
                         'pgf.preamble': (r'\usepackage{amsmath,bm}'
                                          r'\pdfinfoomitdate=1\pdftrailerid{}'
                                          r'\pdfsuppressptexinfo=15'), 'font.size': 8,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42})


def new_figure(probability_labels=False):
    """Feste Druckgeometrie: zwei gleich große, unverzerrte Plotflächen.

    Die Höhe wird aus Breite und Figure-Seitenverhältnis abgeleitet.
    Kein automatisches Layout verändert beim Export diese Positionen.
    """
    figure_height = 3.55 if probability_labels else 3.35
    bottom = 0.23 if probability_labels else 0.18
    fig = plt.figure(figsize=(6.3, figure_height))
    # GridSpec steuert den Zwischenraum; beide Plotflächen bleiben quadratisch.
    left, right, gap = 0.075, 0.88, 0.15
    width = (right - left - gap) / 2
    height = width * 6.3 / figure_height
    grid = fig.add_gridspec(1, 2, left=left, right=right,
                           bottom=bottom, top=bottom + height,
                           wspace=gap / width)
    axes = [fig.add_subplot(grid[0, column]) for column in range(2)]
    return fig, axes


def draw_panel(ax, points, y, w, b, limits, labels, title, transform=None):
    """Punkte, sigmoid(Score) und Nulllinie desselben Scores zeichnen."""
    grid = np.linspace(*limits, 400)
    xx, yy = np.meshgrid(grid, grid)
    objects = np.stack([xx, yy], axis=-1)
    features = objects if transform is None else transform(objects)
    scores = features @ w + b
    # Identische Wahrscheinlichkeitsskala in allen vier Teilabbildungen.
    cmap = LinearSegmentedColormap.from_list('probability', ['#e7eff5', '#657d91', '#fff3c4'])
    image = ax.pcolormesh(xx, yy, sigmoid(scores), cmap=cmap,
                          vmin=0, vmax=1, shading='auto', rasterized=True)
    ax.contour(xx, yy, scores, levels=[0], colors=['#b62231'], linewidths=1.5)
    ax.scatter(*points[y == 0].T, marker='x', c='#202b35', s=18, linewidths=1)
    ax.scatter(*points[y == 1].T, marker='*', c='#f4c44e', edgecolors='#202b35',
               s=55, linewidths=0.7)
    ax.set(xlim=limits, ylim=limits, xlabel=labels[0], ylabel=labels[1], title=title)
    ax.set_aspect('equal')
    return image


def finish(fig, axes, image, name, colorbar_label, exam=False):
    # In Achsenkoordinaten verankert: exakt y=0 bis y=1 der Plotfläche,
    # unabhängig von Titeln, Beschriftungen, Renderer und Exportformat.
    # Automatische Labelposition außerhalb der Ticklabels, kein fester Versatz.
    axes[1].yaxis.labelpad = 6
    from matplotlib.transforms import Bbox
    def colorbar_position(_ax, _renderer):
        box = axes[1].get_position()
        return Bbox.from_bounds(box.x1 + 0.045 * box.width, box.y0,
                                0.04 * box.width, box.height)
    colorbar_axes = fig.add_axes([0, 0, 0.01, 0.01], axes_locator=colorbar_position)
    bar = fig.colorbar(image, cax=colorbar_axes, ticks=[0, .5, 1])
    bar.set_label(colorbar_label)
    handles = [Line2D([], [], color='#b62231', label=r'Entscheidungsrand ($p=0{,}5$)'),
               Line2D([], [], marker='x', color='#202b35', ls='',
                      label='Nicht bestanden' if exam else 'Klasse 0'),
               Line2D([], [], marker='*', color='#202b35', markerfacecolor='#f4c44e',
                      markersize=9, ls='', label='Prüfung bestanden' if exam else 'Klasse 1')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.50, 0.005), ncol=3,
               fontsize=6.5, frameon=False, columnspacing=1.2, handletextpad=0.5,
               borderaxespad=0)
    OUTPUT.mkdir(exist_ok=True)
    # Die Druckdatei wird tatsächlich von LaTeX gesetzt (PGF-Backend).
    # Feste PDF-Metadaten vermeiden zeitabhängige Export-Unterschiede.
    fig.savefig(OUTPUT / f'{name}.pdf', backend='pgf',
                metadata={'CreationDate': None, 'ModDate': None})
    # Vorschau direkt aus der gesetzten PDF, keine abweichenden Ersatzschriften.
    import subprocess
    subprocess.run(['pdftoppm', '-singlefile', '-r', '200', '-png',
                    str(OUTPUT / f'{name}.pdf'), str(OUTPUT / name)], check=True)


def save_results(name, x, y, w, b, training_features, steps):
    OUTPUT.mkdir(exist_ok=True)
    np.savetxt(OUTPUT / f'{name}_data.csv', np.column_stack([x, y]), delimiter=',',
               header='x1,x2,y', comments='', fmt='%.17g')
    result = {'w': w.tolist(), 'b': b, 'training_accuracy': accuracy(training_features, y, w, b),
              'steps': steps, 'learning_rate': 0.3, 'n': len(y)}
    (OUTPUT / f'{name}_model.json').write_text(json.dumps(result, indent=2) + '\n')
    print(name, json.dumps(result))


def interactive(x, y, w, b, quadratic=False):
    """Optional: Parameter verändern, ohne Druckausgaben zu überschreiben.

    Benötigt ein GUI-Backend von Matplotlib (z.B. Tk oder Qt).
    Gibt Figure, Regler und Callback auch für einen nichtinteraktiven Test zurück.
    """
    from matplotlib.widgets import Slider
    fig, ax = plt.subplots(figsize=(6.3, 5))
    fig.subplots_adjust(bottom=0.30)
    sliders = []
    for i, (label, value) in enumerate(zip(['w1', 'w2', 'b'], [*w, b])):
        span = max(2 * abs(value), 0.1)
        slider_ax = fig.add_axes([.23, .18 - .055*i, .62, .025])
        sliders.append(Slider(slider_ax, label, -span, span, valinit=value))
    def update(_=None):
        weights = np.array([sliders[0].val, sliders[1].val])
        bias = sliders[2].val
        ax.clear()
        features = phi(x) if quadratic else x
        acc = accuracy(features, y, weights, bias)
        draw_panel(ax, x, y, weights, bias, (-15, 15), ('$x_1$', '$x_2$'),
                   f'Freie Parameterwahl: {acc:.0%} Trainingsgenauigkeit',
                   transform=phi if quadratic else None)
        fig.canvas.draw_idle()
    for slider in sliders:
        slider.on_changed(update)
    update()
    return fig, sliders, update
