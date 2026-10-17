"""Static, shareable scientific plots and a self-contained HTML report."""
import base64
import csv
import html
from pathlib import Path
import numpy as np


def write_report(folder, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = Path(folder)
    dns, generated = result['dns'], result['generated']
    groups = [('Held-out DNS', dns), ('Generated', generated)]
    images = []

    def save(figure, name, title):
        figure.tight_layout()
        figure.savefig(folder/(name+'.png'), dpi=150)
        figure.savefig(folder/(name+'.pdf'))
        plt.close(figure)
        encoded = base64.b64encode((folder/(name+'.png')).read_bytes()).decode()
        images.append(f'<h2>{html.escape(title)}</h2><img alt="{html.escape(title)}" src="data:image/png;base64,{encoded}">')

    fig, ax = plt.subplots(figsize=(7,4))
    for label, values in groups:
        k = np.asarray(values['spectrum_k'])
        energy = np.asarray(values['spectrum_shell_energy'])
        positive = (k > 0) & (energy > 0)
        ax.loglog(k[positive], energy[positive], label=label)
    ax.set(xlabel='Angular wavenumber (coordinate units⁻¹)', ylabel='Kinetic energy per shell',
           title='Mean energy spectrum (zero mode omitted from plot)')
    ax.legend()
    save(fig, 'energy-spectrum', 'Energy spectra')

    fig, ax = plt.subplots(figsize=(6,4))
    ax.boxplot([dns['kinetic_energy']['values'], generated['kinetic_energy']['values']])
    ax.set_xticks([1,2], ['Held-out DNS', 'Generated'])
    ax.set(ylabel='Mean kinetic energy per unit mass', title='Snapshot energy distribution')
    save(fig, 'kinetic-energy', 'Kinetic energy across snapshots')

    fig, axes = plt.subplots(1,3,figsize=(12,3.5))
    edges = np.asarray(dns['pdf_edges'])
    center = .5*(edges[:-1]+edges[1:])
    for i, ax in enumerate(axes):
        for label, values in groups:
            ax.plot(center, values['normalized_velocity_pdf'][i], label=label)
        ax.set(xlabel=f"({('u','v','w')[i]} − training mean) / training std", ylabel='Probability density')
    axes[0].legend()
    save(fig, 'velocity-pdfs', 'Velocity component distributions')

    fig, axes = plt.subplots(1,2,figsize=(9,4))
    for ax, key, label in zip(axes, ('divergence_rms','relative_divergence'),
                               ('RMS divergence (velocity / coordinate units)','RMS divergence / RMS velocity gradient')):
        ax.boxplot([dns[key]['values'],generated[key]['values']])
        ax.set_xticks([1,2],['Held-out DNS','Generated'])
        ax.set_ylabel(label)
    save(fig, 'divergence', 'Divergence diagnostics')

    with open(folder/'summary.csv','w',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(['group','snapshot_count','mean_kinetic_energy','std_kinetic_energy','mean_divergence_rms','mean_relative_divergence'])
        for label, values in groups:
            writer.writerow([label,values['count'],values['kinetic_energy']['mean'],values['kinetic_energy']['std'],
                             values['divergence_rms']['mean'],values['relative_divergence']['mean']])
    with open(folder/'spectra.csv','w',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(['angular_wavenumber','dns_mean_shell_energy','generated_mean_shell_energy'])
        writer.writerows(zip(dns['spectrum_k'],dns['spectrum_shell_energy'],generated['spectrum_shell_energy']))
    comparison = result['comparison']
    lines = [f"DNS snapshots: {dns['count']}; generated snapshots: {generated['count']}.",
             f"Test flow-matching loss: {result['test_flow_loss']['mean_loss']:.6g}.",
             f"Relative bias of mean kinetic energy: {comparison['mean_kinetic_energy_relative_bias']:.3%}.",
             f"Relative L1 spectrum discrepancy (excluding mean mode): {comparison['spectrum_relative_l1_excluding_mean']:.6g}."]
    reconstruction = result.get('test_autoencoder_reconstruction')
    if reconstruction:
        lines.append(f"First test volume autoencoder reconstruction: normalized MSE {reconstruction['normalized_mse']:.6g}; physical MSE {reconstruction['physical_mse']:.6g}.")
    contents = ['<!doctype html><html><head><meta charset="utf-8"><title>3D LapFlow evaluation</title>',
                '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px}img{max-width:100%}li{margin:8px 0}</style>',
                '</head><body><h1>3D LapFlow: held-out statistical evaluation</h1>',
                '<p>This compares unconditional generated volumes with held-out DNS. It is not a forecasting evaluation.</p>',
                '<ul>'+''.join('<li>'+html.escape(line)+'</li>' for line in lines)+'</ul>',
                *images, '<h2>Interpretation limits</h2><ul>',
                *['<li>'+html.escape(line)+'</li>' for line in result['limitations']], '</ul></body></html>']
    (folder/'report.html').write_text('\n'.join(contents))
