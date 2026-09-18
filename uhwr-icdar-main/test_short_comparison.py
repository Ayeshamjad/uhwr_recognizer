"""Dependency-light preprocessing/selection tests of the comparison script."""
import ast
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

source=ast.parse((Path(__file__).parent/'compare_short_htr.py').read_text())
functions=ast.Module(body=[n for n in source.body if isinstance(n,ast.FunctionDef)
                          and n.name in ('modified_pixels','select_samples')],type_ignores=[])


class Tests(unittest.TestCase):
    def setUp(self):
        self.namespace={'np':np,'Image':Image,'Path':Path,'json':json,'csv':csv,'math':__import__('math'),
                        'torch':SimpleNamespace(from_numpy=lambda x:x,tensor=np.array)}
        exec(compile(functions,'comparison_functions','exec'),self.namespace)

    def test_width_preserved_and_padding_white(self):
        pixels,width=self.namespace['modified_pixels'](Image.new('L',(923,64),255))
        self.assertEqual(pixels.shape,(1,1,928,64))
        self.assertEqual(width.tolist(),[923])
        self.assertTrue(np.all(pixels==0))

    def test_vertical_flip_not_horizontal_reversal(self):
        array=np.full((64,32),255,dtype=np.uint8)
        array[0,0]=0
        pixels,_=self.namespace['modified_pixels'](Image.fromarray(array))
        self.assertEqual(pixels[0,0,0,63],1)
        self.assertEqual(pixels[0,0,31,63],0)

    def test_height_normalization_keeps_aspect_ratio(self):
        pixels,width=self.namespace['modified_pixels'](Image.new('L',(100,32),255))
        self.assertEqual(width.tolist(),[200])
        self.assertEqual(pixels.shape,(1,1,200,64))

    def test_invalid_width_rejected(self):
        with self.assertRaises(ValueError):
            self.namespace['modified_pixels'](Image.new('L',(9000,64),255))

    def test_csv_paths_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            Image.new('L',(128,64),255).save(root/'word.png')
            with (root/'labels.csv').open('w',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=['file_name','text'])
                writer.writeheader();writer.writerow({'file_name':'word.png','text':'اردو'})
            args=SimpleNamespace(samples_csv=str(root/'labels.csv'),image_root=None,count=20)
            rows=self.namespace['select_samples'](args)
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['text'],'اردو')
            self.assertEqual(rows[0]['image'].size,(128,64))


if __name__=='__main__':
    unittest.main()
