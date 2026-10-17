import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from netCDF4 import Dataset
from lapflow.autoencoder import Autoencoder3D, periodic_window, core, encode_tiled, decode_tiled
from lapflow.config import validate
from lapflow.data import read_window, discover
from lapflow.flow import pyramid, reconstruct, path, noise_pyramid, sample, loss
from lapflow.model import LapFlow3D, causal_attention
from lapflow.train import prepare, train_ae, cache_latents, train_flow, generate
from lapflow.evaluate import reconstruct_volume


def tiny_config(root):
    return {'data_root': str(root), 'grid': 16, 'channels': ['u','v','w'],
            'splits': {'train': [1,1], 'val': [2,2], 'test': [3,3]}, 'seed': 42,
            'autoencoder': {'widths': [4], 'latent_channels': 2, 'tile': 8, 'halo': 4},
            'model': {'width': 12, 'depth': 1, 'heads': 2, 'patch': 2, 'levels': 3,
                      'query_chunk': 8, 'checkpoint': True},
            'flow': {'starts': [0.,1/3,2/3], 'steps_per_stage': 2},
            'training': {'ae_steps': 2, 'flow_steps': 2, 'lr': .001, 'save_every': 1,
                         'stats_crops': 2, 'batch_size': 1, 'amp': False, 'validation_samples': 1}}


def write_fixture(root):
    rng = np.random.default_rng(5)
    for file_id in (1,2,3):
        with Dataset(Path(root)/f'fluid_{file_id}.nc','w') as ds:
            for axis, size in [('time',1),('x',16),('y',16),('z',16)]:
                ds.createDimension(axis,size)
                ds.createVariable(axis,'f4',(axis,))[:] = [file_id] if axis=='time' else np.arange(size)
            for channel in ('u','v','w'):
                ds.createVariable(channel,'f4',('time','x','y','z'))[:] = rng.normal(size=(1,16,16,16))


class MathTests(unittest.TestCase):
    def test_pyramid_exact_reconstruction(self):
        x = torch.randn(2,3,16,16,16)
        torch.testing.assert_close(reconstruct(pyramid(x)), x, atol=5e-7, rtol=1e-6)
        for residual in pyramid(x)[1:]:
            torch.testing.assert_close(torch.nn.functional.avg_pool3d(residual,2),
                                       torch.zeros_like(residual[...,::2,::2,::2]),atol=2e-7,rtol=0)

    def test_flow_path_endpoints_and_derivative(self):
        clean = pyramid(torch.randn(2,3,8,8,8,dtype=torch.float64))
        noise = [torch.randn_like(x) for x in clean]
        starts = [0.,1/3,2/3]
        for active in (1,2,3):
            t = torch.full((2,),starts[active-1],dtype=torch.float64)
            x,u = path(clean,noise,t,starts,active)
            torch.testing.assert_close(x[-1],(1-t[0])*noise[active-1])
            end,_ = path(clean,noise,torch.ones_like(t),starts,active)
            torch.testing.assert_close(end[-1],clean[active-1])
            later,_ = path(clean,noise,t+1e-5,starts,active)
            for a,b,v in zip(x,later,u):
                torch.testing.assert_close((b-a)/1e-5,v,rtol=1e-8,atol=1e-8)

    def test_attention_matches_dense_mask_and_gradients(self):
        torch.manual_seed(1)
        lengths = [2,3,5]
        q,k,v = [[torch.randn(1,2,n,4,dtype=torch.float64,requires_grad=True) for n in lengths] for _ in range(3)]
        actual = torch.cat(causal_attention(q,k,v,chunk=2,scale=.125),-2)
        labels = torch.repeat_interleave(torch.arange(3),torch.tensor(lengths))
        allowed = labels[:,None] >= labels[None,:]
        scores = torch.cat(q,-2) @ torch.cat(k,-2).transpose(-1,-2)*.125
        expected = scores.masked_fill(~allowed,float('-inf')).softmax(-1) @ torch.cat(v,-2)
        torch.testing.assert_close(actual,expected)
        ga = torch.autograd.grad(actual.square().sum(),q+k+v,retain_graph=True)
        gb = torch.autograd.grad(expected.square().sum(),q+k+v)
        for a,b in zip(ga,gb):torch.testing.assert_close(a,b)

    def test_sampler_constant_target_reaches_clean_pyramid(self):
        clean = pyramid(torch.randn(1,2,8,8,8))
        starts = [0.,1/3,2/3]
        generator = torch.Generator().manual_seed(9)
        noise = noise_pyramid(torch.randn(1,2,8,8,8,generator=generator),3,generator)
        seen=[]
        def oracle(states,t):
            seen.append(len(states))
            return tuple(clean[i]/(1-starts[i])-noise[i] for i in range(len(states)))
        result = sample(oracle,(1,2,8,8,8),starts,4,'cpu',torch.Generator().manual_seed(9))
        torch.testing.assert_close(result,reconstruct(clean),atol=2e-6,rtol=2e-6)
        self.assertEqual(seen,[1]*8+[2]*8+[3]*8)

    def test_checkpoint_outputs_and_gradients_match(self):
        torch.manual_seed(2)
        model = LapFlow3D(2,width=12,depth=2,heads=2,patch=2,levels=3,query_chunk=4)
        # Nonzero gates/heads exercise attention and checkpointed gradients.
        for p in model.parameters():
            torch.nn.init.normal_(p, std=.08)
        reference = copy.deepcopy(model)
        reference.checkpoint = False
        states = tuple(pyramid(torch.randn(1,2,8,8,8)))
        t = torch.tensor([.8])
        a,b = model(states,t),reference(states,t)
        for x,y in zip(a,b):torch.testing.assert_close(x,y)
        sum(x.square().mean() for x in a).backward()
        sum(x.square().mean() for x in b).backward()
        for p,q in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(p.grad,q.grad)

    def test_fine_tokens_cannot_change_coarse_output(self):
        model = LapFlow3D(2,width=12,depth=1,heads=2,patch=2,levels=3)
        for p in model.parameters():torch.nn.init.normal_(p,std=.1)
        states = pyramid(torch.randn(1,2,8,8,8))
        altered = [states[0],states[1]+5,states[2]-3]
        torch.testing.assert_close(model(states,torch.tensor([.8]))[0],model(altered,torch.tensor([.8]))[0])


class TileTests(unittest.TestCase):
    def test_tiled_matches_periodic_full_encoder_and_decoder(self):
        torch.manual_seed(4)
        for widths,grid,tile,halo in [([4],16,8,4),([4,8],32,16,8),([8,16,24,32],64,32,32)]:
            model = Autoencoder3D(widths,2).eval()
            x = torch.randn(1,3,grid,grid,grid)
            reader = lambda origin,size:periodic_window(x,origin,size)
            tiled = encode_tiled(model,reader,grid,tile,halo,'cpu')
            padded = periodic_window(x,(-halo,)*3,grid+2*halo)
            with torch.no_grad():
                full = core(model.encoder(padded),halo//model.factor,grid//model.factor)
                torch.testing.assert_close(tiled,full,atol=2e-6,rtol=1e-5)
                decoded = decode_tiled(model,tiled,tile,halo,'cpu')
                expanded = periodic_window(full,(-halo//model.factor,)*3,(grid+2*halo)//model.factor)
                reference = core(model.decoder(expanded),halo,grid)
                torch.testing.assert_close(decoded,reference,atol=2e-6,rtol=1e-5)
                # Training crop reconstruction and deployed tiled reconstruction agree on core.
                crop = model(reader((-halo,)*3,tile+2*halo))
                torch.testing.assert_close(core(crop,halo,tile),decoded[...,:tile,:tile,:tile],atol=2e-6,rtol=1e-5)

    def test_netcdf_periodic_read_and_header(self):
        with tempfile.TemporaryDirectory() as root:
            write_fixture(root)
            c=tiny_config(root)
            manifest=discover(c)
            record=manifest['records']['train'][0]
            full=read_window(record,(0,0,0),16)
            partial=read_window(record,(-4,14,7),12)
            torch.testing.assert_close(partial,periodic_window(full,(-4,14,7),12))
            self.assertEqual(len(manifest['records']['train']),1)

    def test_configuration_rejects_overlap_and_invalid_tiles(self):
        c=tiny_config('.')
        validate(c)
        c['splits']['val']=[1,1]
        with self.assertRaises(ValueError):validate(c)
        c=tiny_config('.')
        c['autoencoder']['halo']=1
        with self.assertRaises(ValueError):validate(c)


class PipelineTests(unittest.TestCase):
    def test_complete_pipeline_and_resume_provenance(self):
        with tempfile.TemporaryDirectory() as root:
            write_fixture(root)
            c=tiny_config(root)
            run=Path(root)/'run'
            device=torch.device('cpu')
            torch.manual_seed(c['seed'])
            prepare(c,run)
            train_ae(c,run,device)
            with self.assertRaises(FileExistsError):train_ae(c,run,device)
            train_ae(c,run,device,resume=True)
            reconstruct_volume(c,run,device)
            self.assertTrue((run/'reconstruction-val-0.json').exists())
            cache_latents(c,run,device)
            cache_latents(c,run,device)  # Existing entries are validated and reused.
            train_flow(c,run,device)
            train_flow(c,run,device,resume=True)
            generate(c,run,device,8)
            with Dataset(run/'sample-8.nc') as ds:
                self.assertEqual(ds['u'].shape,(16,16,16))
                self.assertTrue(np.isfinite(ds['u'][:]).all())
                self.assertIn('Unconditional',ds.description)
            with self.assertRaises(FileExistsError):generate(c,run,device,8)
            altered=copy.deepcopy(c)
            altered['seed']+=1
            with self.assertRaises(ValueError):train_flow(altered,run,device,resume=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
