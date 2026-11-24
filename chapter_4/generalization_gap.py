"""Schematische Lernfehler. Aufruf: python chapter_4/generalization_gap.py"""
from risk_common import (risk_components, training_risk, schematic_curves, setup_figure,
                         plot_generalization, save_figure, ORANGE)


def main():
    curves = schematic_curves()
    fig, ax = setup_figure(curves)
    ax.plot(curves['capacity'], curves['training'], color=ORANGE, lw=1.8, ls='--',
            label=r'$\mathcal J_{\mathcal S}(f_{\mathcal S})$ (Trainingsfehler)')
    plot_generalization(ax, curves)

    # Generalisierungslücke: wahres Risiko minus Trainingsrisiko, nicht Schätzfehler.
    position = .74
    base, app, est = risk_components(position)
    training = float(training_risk(position))
    generalization = float(base + app + est)
    ax.annotate('', xy=(position, generalization), xytext=(position, training),
                arrowprops=dict(arrowstyle='<->', color='#555555', lw=1, shrinkA=0, shrinkB=0))
    ax.text(position+.025, (training + generalization)/2,
            'Generalisierungs-\nlücke', ha='left', va='center', fontsize=8)
    save_figure(fig, ax, 'generalization_gap', curves, dict(gap_capacity=position))


if __name__ == '__main__':
    main()
