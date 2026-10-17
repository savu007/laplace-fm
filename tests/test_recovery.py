"""Regression tests for periodic I/O cost and preservation of completed updates."""
import contextlib
import io
import json
import tempfile
import unittest
from itertools import product
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from lapflow.autoencoder import periodic_window
from lapflow.data import read_window, snapshot_reader, normalize, atomic_json
from lapflow.pipeline import import_prepared, artifact_stamp
from lapflow.runtime import seed_all
from lapflow.train import prepare, train_ae, cache_latents, train_flow
from tests.test_core import tiny_config, write_fixture


class ReaderTests(unittest.TestCase):
    def test_wrapped_reads_are_contiguous_and_bounded(self):
        data = np.arange(16**3, dtype=np.float32).reshape(1,16,16,16)
        calls = []
        class Variable:
            shape = data.shape
            def __getitem__(self, key):
                self_key = key
                if not all(isinstance(x, slice) for x in key[1:]):
                    raise AssertionError('NetCDF fancy indexing is forbidden')
                calls.append(self_key)
                return data[key]
        class File:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def set_auto_mask(self, value): pass
            def __getitem__(self, key): return Variable()
        with patch('lapflow.data.Dataset', return_value=File()):
            result = read_window({'file':'unused','index':0}, (-4,14,15), 12)
        self.assertEqual(len(calls),24)
        full = torch.from_numpy(np.repeat(data[:,None],3,axis=1))
        torch.testing.assert_close(result,periodic_window(full,(-4,14,15),12))

    def test_all_boundaries_and_multiple_wraps_match_in_memory(self):
        with tempfile.TemporaryDirectory() as root:
            write_fixture(root)
            record = {'file':str(Path(root)/'fluid_1.nc'),'index':0}
            full = read_window(record,(0,0,0),16)
            stats = {'mean':[1.,2.,3.],'std':[2.,3.,4.]}
            cached = snapshot_reader(record,16,stats)
            for origin in product((0,-4,14),repeat=3):
                expected = periodic_window(full,origin,12)
                torch.testing.assert_close(read_window(record,origin,12),expected,rtol=0,atol=0)
                torch.testing.assert_close(cached(origin,12),normalize(expected,stats),rtol=0,atol=0)
            torch.testing.assert_close(read_window(record,(-35,50,14),35),
                                       periodic_window(full,(-35,50,14),35),rtol=0,atol=0)


class RecoveryTests(unittest.TestCase):
    def test_checkpoint_before_validation_and_exact_resume(self):
        with tempfile.TemporaryDirectory() as root, contextlib.redirect_stdout(io.StringIO()):
            write_fixture(root)
            c=tiny_config(root)
            # Fail before best.pt exists, both at intermediate and final updates.
            for final_step in (1,2):
                c['training']['ae_steps']=final_step
                c['training']['flow_steps']=final_step
                for stage, trainer, validator in [('autoencoder',train_ae,'validation_ae'),('flow',train_flow,'validation_flow')]:
                    runs=[]
                    for label in ('reference','interrupted'):
                        run=Path(root)/f'{stage}-{final_step}-{label}'
                        prepare(c,run)
                        if stage=='flow':
                            seed_all(10)
                            train_ae(c,run,torch.device('cpu'))
                            cache_latents(c,run,torch.device('cpu'))
                        seed_all(11)
                        if label=='interrupted':
                            with patch('lapflow.train.'+validator,side_effect=RuntimeError('validation interrupted')):
                                with self.assertRaisesRegex(RuntimeError,'validation interrupted'):
                                    trainer(c,run,torch.device('cpu'))
                            saved=torch.load(run/stage/'last.pt',weights_only=False)
                            self.assertEqual(saved['step'],1)
                            self.assertFalse((run/stage/'best.pt').exists())
                            trainer(c,run,torch.device('cpu'),resume=True)
                        else:
                            trainer(c,run,torch.device('cpu'))
                        runs.append(run)
                    a,b=[torch.load(run/stage/'last.pt',weights_only=False) for run in runs]
                    self.assertEqual(a['step'],final_step)
                    self.assertEqual(b['step'],final_step)
                    for name in a['model']:
                        torch.testing.assert_close(a['model'][name],b['model'][name],rtol=0,atol=0)
                    self.assertEqual(a['best'],b['best'])

    def test_prepared_import_checks_config_and_artifacts(self):
        with tempfile.TemporaryDirectory() as root, contextlib.redirect_stdout(io.StringIO()):
            write_fixture(root)
            c=tiny_config(root)
            source=Path(root)/'source'; target=Path(root)/'target'
            prepare(c,source); target.mkdir()
            atomic_json(source/'pipeline-status.json',{'plan':{'source_hash':'old-code'},'stages':{
                'prepare':{'state':'complete','artifacts':{name:artifact_stamp(source/name)
                            for name in ('dataset.json','config.json')}}}})
            import_prepared(c,target,source)
            import_prepared(c,target,source)
            self.assertEqual(json.loads((source/'dataset.json').read_text()),json.loads((target/'dataset.json').read_text()))
            changed=dict(c,seed=999)
            with self.assertRaisesRegex(ValueError,'configuration differs'):
                import_prepared(changed,target,source)
            with open(source/'dataset.json','a') as handle: handle.write(' ')
            with self.assertRaisesRegex(ValueError,'artifact changed'):
                import_prepared(c,target,source)
