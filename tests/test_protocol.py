import unittest,json,tempfile,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from metrics import confusion,row_metrics,aggregate,decode_mask,union_confusions
from features import letterbox,sha,verify_encoder_source
from PIL import Image
import markers
from calibrate_clip import choose
from predict import binding
from calibrate_rgb import response
from extract_source import indices

class ProtocolTests(unittest.TestCase):
    def test_all_foreground(self):
        r=row_metrics(confusion(np.ones((2,3)),np.ones((2,3))));self.assertEqual(r['fgiou'],1);self.assertEqual(r['tp'],6)
    def test_source_identity_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):verify_encoder_source(d)
    def test_rgb_exact_coefficients(self):
        v=response(np.array([[[255,255,0],[0,255,0],[0,0,255],[255,255,255],[0,0,0]]],np.uint8))
        np.testing.assert_allclose(v,[[.9666666667,.2333333333,-.8666666667,.1,0]],rtol=1e-6,atol=1e-7)
    def test_stable_source_sampling(self):
        rows=[dict(image='train/a.jpg',legacy_sampling=False),dict(image='train/b.jpg',legacy_sampling=False)]
        a=indices(rows);b=indices(rows[::-1]);np.testing.assert_array_equal(a,b[::-1]);self.assertEqual(len(set(a[0])),2000)
    def test_assignment_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'m.npz').write_bytes(b'density');(p/'encoder').write_bytes(b'e')
            assignment=dict(model_sha256=sha(p/'m.npz'),selected_component=2)
            (p/'a.json').write_text(json.dumps(assignment))
            b=dict(model_file='m.npz',model_sha256=sha(p/'m.npz'),encoder_sha256=sha(p/'encoder'),assignment_file='a.json',assignment_record_sha256=sha(p/'a.json'),foreground_components=[1])
            (p/'model.json').write_text(json.dumps(b))
            with self.assertRaisesRegex(ValueError,'Semantic mapping'):binding(p/'model.json',p/'encoder')
            assignment['model_sha256']='different';(p/'a.json').write_text(json.dumps(assignment));b['assignment_record_sha256']=sha(p/'a.json');(p/'model.json').write_text(json.dumps(b))
            with self.assertRaisesRegex(ValueError,'another density'):binding(p/'model.json',p/'encoder')
    def test_native_support(self):
        c=confusion(np.array([[1,1],[0,0]]),np.array([[1,0],[1,0]]));self.assertEqual(c,dict(tp=1,fp=1,fn=1,tn=1))
    def test_ignore(self):
        c=confusion(np.array([[1,0],[1,1]]),np.array([[1,99],[0,99]]),np.array([[1,0],[1,0]],bool));self.assertEqual(c,dict(tp=1,fp=1,fn=0,tn=0))
    def test_empty_support_rejected(self):
        with self.assertRaises(ValueError):confusion(np.zeros((2,2)),np.zeros((2,2)),np.zeros((2,2)))
    def test_empty_is_not_perfect_iou(self):
        r=row_metrics(dict(tp=0,fp=0,fn=0,tn=4));self.assertIsNone(r['fgiou']);self.assertIsNone(aggregate([r])['mean_image_FGIoU'])
    def test_empty_false_positive(self):
        r=row_metrics(dict(tp=0,fp=2,fn=0,tn=2));self.assertEqual(r['fgiou'],0);self.assertEqual(aggregate([r])['visible_area_MAE_pp'],50)
    def test_non_square_letterbox(self):
        a,p=letterbox(Image.fromarray(np.full((300,600,3),100,np.uint8)));self.assertEqual(a.shape,(1120,1120,3));self.assertEqual(p,(280,0,560,1120,300,600));self.assertTrue((a[:280]==0).all())
    def test_red_palette(self):self.assertEqual(decode_mask(np.array([[[255,0,0],[0,0,0]]],np.uint8)).tolist(),[[1,0]])
    def test_bad_palette(self):
        with self.assertRaises(ValueError):decode_mask(np.array([[[3,4,5]]],np.uint8))
    def test_union(self):
        labels=np.array([[0,1],[2,2]]);gt=np.array([[1,0],[1,0]]);c=[confusion(labels==k,gt) for k in range(3)]
        self.assertEqual(union_confusions(c,[0,2]),confusion(np.isin(labels,[0,2]),gt))
    def test_union_invalid_partition(self):
        with self.assertRaises(ValueError):union_confusions([dict(tp=1,fp=1,fn=1,tn=1)]*3,[0])
    def test_semantic_permutation(self):
        s=np.array([[.1,.2,.5],[.3,.1,.6]]);p=np.array([2,0,1]);self.assertEqual(p[choose(s[:,p])['selected_component']],choose(s)['selected_component'])
    def test_semantic_tie(self):self.assertEqual(choose([[1,1,0]])['selected_component'],0)
    def test_semantic_invalid(self):
        with self.assertRaises(ValueError):choose([[1,float('nan'),0]])
    def test_identity_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'m.npz').write_bytes(b'wrong');(p/'encoder').write_bytes(b'e');(p/'model.json').write_text(json.dumps(dict(model_file='m.npz',model_sha256='invalid')))
            with self.assertRaises(ValueError):binding(p/'model.json',p/'encoder')
    def test_fallback_and_min_area(self):
        p=np.zeros((12,12),np.float32);p[2:4,2:4]=.8;p[8,8]=.9
        cfg=dict(method='erosion_core',low_threshold=.35,erosion_iterations=3,seed_min_area=4)
        self.assertEqual(markers.marker_count(p,cfg),1)
    def test_empty_count(self):
        self.assertEqual(markers.marker_count(np.zeros((12,12)),dict(method='erosion_core',low_threshold=.35,erosion_iterations=3,seed_min_area=4)),0)
    def test_grid(self):self.assertEqual(len(markers.configurations()),144)

if __name__=='__main__':unittest.main()
