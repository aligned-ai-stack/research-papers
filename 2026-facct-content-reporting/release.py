"""Create a release archive from an explicit reviewed file allowlist."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
FILES = ['README.md', '.gitignore', 'run.py', 'plots.py', 'requirements.txt', 'LICENSE-CODE', 'LICENSE-DATA', 'CITATION.cff', 'deposit-metadata.json', 'tests/test_public_workflow.py', 'analysis/H1.py', 'analysis/H2.py', 'analysis/H3.py', 'analysis/H4.py', 'analysis/H5.py', 'analysis/H6.py', 'data/accuracy.csv', 'data/distance.csv', 'data/explanation_quality.csv', 'data/input_dictionary.csv', 'data/models/H1_descriptives_by_condition.csv', 'data/models/H1_descriptives_by_condition_errorType.csv', 'data/models/H1_exploratory_GLM_covariates_HC1_tidy.csv', 'data/models/H1_OR_by_errorType.csv', 'data/models/H1_primary_GLM_HC1_tidy.csv', 'data/models/H1_robustA_GLM_cluster_tidy.csv', 'data/models/H2_robust_covariates_tidy.csv', 'data/models/H2a_companion_participant_robust.csv', 'data/models/H2a_primary_prereg_no_caseFE_tidy.csv', 'data/models/H2a_robust_design_caseFE_tidy.csv', 'data/models/H2a_robust_order_FE_tidy.csv', 'data/models/H2b_primary_prereg_no_caseFE_tidy.csv', 'data/models/H2b_robust_design_caseFE_tidy.csv', 'data/models/H3b_timing_specifications.csv', 'data/models/H4_OLS_contrasts.csv', 'data/models/H4_OLS_tidy.csv', 'data/models/H5_accuracy_GLM_DiD_tidy.csv', 'data/models/H5_accuracy_margins.csv', 'data/models/H5_accuracy_means_by_pairErrorType_post.csv', 'data/models/H5_descriptives_condition_pairErrorType_post.csv', 'data/models/H5_EoR_FD_OLS_tidy_all.csv', 'data/models/H5_EoR_sensitivity_ordered_logit_tidy_all.csv', 'data/models/H6_overbreadth_GLM_tidy.csv', 'data/provenance.json', 'data/table_dictionary.csv', 'figures/accuracy.png', 'figures/timing_specifications.png', 'release.py']

def package(destination):
    destination = Path(destination).resolve()
    if destination in [(ROOT / name).resolve() for name in FILES]:
        raise ValueError('Archive cannot overwrite a release input')
    manifest = {}
    payload = {}
    for name in FILES:
        path = ROOT / name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f'Missing or unsafe release input: {name}')
        payload[name] = path.read_bytes()
        manifest[name] = hashlib.sha256(payload[name]).hexdigest()
    payload['SHA256SUMS.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in payload.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 4, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)
    return manifest

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    package(parser.parse_args().output)
