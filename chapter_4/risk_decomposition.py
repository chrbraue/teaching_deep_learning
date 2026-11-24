"""Schematische Fehlerzerlegung. Aufruf: python chapter_4/risk_decomposition.py"""
from risk_common import (risk_components, schematic_curves, setup_figure,
                         plot_generalization, save_figure, ORANGE)


def main():
    curves = schematic_curves()
    fig, ax = setup_figure(curves)
    ax.plot(curves['capacity'], curves['minimum_risk'], color='#777777',
            lw=1.8, ls='--', label=r'$\mathcal J_{\mathcal D}(f_{\mathcal D})$')
    ax.plot(curves['capacity'], curves['estimation'], color=ORANGE,
            lw=1.8, ls='-.', label=r'$\varepsilon_{\mathrm{est}}$ (Schätzfehler)')
    plot_generalization(ax, curves)
    ax.axhline(curves['bayes'][0], color='#333333', lw=1, ls=':',
               label=r'$\mathcal J_{\mathcal D}(f_{\mathrm B})$ (Bayes-Risiko)')

    # Die Pfeile bezeichnen Differenzen bei jeweils derselben Kapazität.
    app_capacity = .27
    base, app, _ = risk_components(app_capacity)
    ax.annotate('', xy=(app_capacity, float(base + app)),
                xytext=(app_capacity, float(base)),
                arrowprops=dict(arrowstyle='<->', color='#555555', lw=1, shrinkA=0, shrinkB=0))
    ax.text(app_capacity-.025, float(base + app/2), r'$\varepsilon_{\mathrm{app}}$',
            ha='right', va='center')

    est_capacity = .74
    base, app, est = risk_components(est_capacity)
    lower, upper = float(base + app), float(base + app + est)
    ax.annotate('', xy=(est_capacity, upper), xytext=(est_capacity, lower),
                arrowprops=dict(arrowstyle='<->', color='#555555', lw=1, shrinkA=0, shrinkB=0))
    ax.text(est_capacity-.025, (lower + upper)/2, r'$\varepsilon_{\mathrm{est}}$',
            ha='right', va='center')
    save_figure(fig, ax, 'risk_decomposition', curves,
                dict(approximation_capacity=app_capacity,
                     estimation_capacity=est_capacity))


if __name__ == '__main__':
    main()
