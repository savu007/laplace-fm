import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from lapflow.pipeline import experiment
from lapflow.statistics import volume_statistics, coordinate_spacing
from lapflow.train import generate
from tests.test_core import tiny_config, write_fixture


class PhysicalStatisticsTests(unittest.TestCase):
    def test_shear_wave_parseval_spectrum_and_divergence(self):
        n = 16
        x = torch.arange(n,dtype=torch.float64)*2*math.pi/n
        velocity = torch.zeros(3,n,n,n,dtype=torch.float64)
        velocity[1] = x.sin()[:,None,None]
        stats = volume_statistics(velocity,[2*math.pi/n]*3,{'mean':[0]*3,'std':[1]*3})
        self.assertAlmostEqual(stats['kinetic_energy'],.25,places=12)
        self.assertAlmostEqual(stats['spectrum_shell_energy'][1],.25,places=12)
        self.assertLess(stats['parseval_relative_error'],1e-12)
        self.assertAlmostEqual(stats['divergence_rms'],0.,places=12)
        self.assertAlmostEqual(stats['velocity_gradient_rms'],math.sqrt(.5),places=12)
        self.assertAlmostEqual(np.sum(stats['normalized_velocity_pdf'][1])*.1,1.,places=10)

    def test_compressive_wave_and_reversed_coordinate_orientation(self):
        n = 15
        x = torch.arange(n,dtype=torch.float64)*2*math.pi/n
        velocity = torch.zeros(3,n,n,n,dtype=torch.float64)
        velocity[0] = x.sin()[:,None,None]
        stats = volume_statistics(velocity,[-2*math.pi/n,2*math.pi/n,2*math.pi/n],{'mean':[0]*3,'std':[1]*3})
        self.assertAlmostEqual(stats['divergence_rms'],math.sqrt(.5),places=12)
        self.assertAlmostEqual(stats['relative_divergence'],1.,places=12)
        self.assertLess(stats['parseval_relative_error'],1e-12)

    def test_constant_mean_mode_and_pdf_tails(self):
        velocity = torch.full((3,8,8,8),10.)
        stats = volume_statistics(velocity,[1]*3,{'mean':[0]*3,'std':[1]*3})
        self.assertAlmostEqual(stats['kinetic_energy'],150.)
        self.assertAlmostEqual(stats['spectrum_shell_energy'][0],150.)
        self.assertEqual(stats['pdf_outside_fraction'],[1.,1.,1.])
        self.assertEqual(stats['divergence_rms'],0.)
        self.assertEqual(stats['relative_divergence'],0.)

    def test_nonuniform_coordinates_rejected(self):
        with self.assertRaises(ValueError):
            coordinate_spacing({'x':[0.,1.,3.],'y':[0.,1.,2.],'z':[0.,1.,2.]})


class FullStackTests(unittest.TestCase):
    def test_interruption_resume_and_full_report(self):
        with tempfile.TemporaryDirectory() as root:
            write_fixture(root)
            config = tiny_config(root)
            run = Path(root)/'full-run'
            device = torch.device('cpu')
            def interrupted(*args,**kwargs):
                generate(*args,**kwargs)
                # Simulate loss of the process between atomic NetCDF and JSON completion.
                (run/'sample-700.json').unlink()
                raise RuntimeError('Simulated interruption after generated volume')
            with patch('lapflow.pipeline.generate',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'Simulated interruption'):
                    experiment(config,run,device,samples=2,seed_base=700)
            status = json.loads((run/'pipeline-status.json').read_text())
            self.assertEqual(status['state'],'failed')
            self.assertEqual(status['active_stage'],'sample-700')
            checkpoint_mtime = (run/'flow'/'last.pt').stat().st_mtime_ns
            sample_mtime = (run/'sample-700.nc').stat().st_mtime_ns
            status = experiment(config,run,device,samples=2,seed_base=700)
            self.assertEqual(status['state'],'complete')
            self.assertEqual((run/'sample-700.nc').stat().st_mtime_ns,sample_mtime)
            self.assertTrue((run/'sample-700.json').exists())
            metrics = json.loads((run/'evaluation'/'metrics.json').read_text())
            self.assertEqual(metrics['dns']['count'],1)
            self.assertEqual(metrics['generated']['count'],2)
            self.assertEqual(metrics['test_flow_loss']['snapshots'],1)
            self.assertIn('test_autoencoder_reconstruction',metrics)
            self.assertLess(metrics['dns']['max_parseval_relative_error'],1e-5)
            report = (run/'evaluation'/'report.html').read_text()
            self.assertIn('data:image/png;base64,',report)
            self.assertIn('not a forecasting evaluation',report)
            for name in ('energy-spectrum','kinetic-energy','velocity-pdfs','divergence'):
                for suffix in ('png','pdf'):
                    self.assertGreater((run/'evaluation'/f'{name}.{suffix}').stat().st_size,100)
            experiment(config,run,device,samples=2,seed_base=700)
            self.assertEqual((run/'flow'/'last.pt').stat().st_mtime_ns,checkpoint_mtime)
            with self.assertRaisesRegex(ValueError,'plan changed'):
                experiment(config,run,device,samples=3,seed_base=700)
            # Do not silently accept externally changed completed outputs.
            with open(run/'evaluation'/'summary.csv','a') as handle:
                handle.write('changed\n')
            with self.assertRaisesRegex(ValueError,'artifact is missing or changed'):
                experiment(config,run,device,samples=2,seed_base=700)
            self.assertEqual(json.loads((run/'pipeline-status.json').read_text())['state'],'failed')


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
