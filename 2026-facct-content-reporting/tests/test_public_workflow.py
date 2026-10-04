from pathlib import Path
import hashlib, json, subprocess, sys, tempfile, unittest, zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import release

class PublicWorkflow(unittest.TestCase):
    def test_public_figures_and_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            subprocess.run([sys.executable,'-B',str(ROOT/'plots.py'),'--data',str(ROOT/'data'),
                            '--output',str(tmp/'figures')],check=True)
            self.assertEqual(len(list((tmp/'figures').glob('*.png'))),2)
            release.package(tmp/'release.zip')
            with zipfile.ZipFile(tmp/'release.zip') as archive:
                self.assertEqual(set(archive.namelist()),set(release.FILES)|{'SHA256SUMS.json'})
                self.assertFalse(any('/review/' in '/'+p or '/local/' in '/'+p for p in archive.namelist()))
                for path,digest in json.loads(archive.read('SHA256SUMS.json')).items():
                    self.assertEqual(hashlib.sha256(archive.read(path)).hexdigest(),digest)
    def test_missing_input_fails_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            result=subprocess.run([sys.executable,'-B',str(ROOT/'run.py'),'--data',str(tmp/'absent'),
                                   '--output',str(tmp/'output')],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertFalse((tmp/'output').exists())
            self.assertIn('Missing required input',result.stderr)
