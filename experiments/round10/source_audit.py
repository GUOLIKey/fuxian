"""Read-only paper/source audit executed on the remote server."""
import hashlib
from pathlib import Path
from pypdf import PdfReader

ROOT = Path('/root/autodl-tmp/fedvade_project')
OUT = ROOT / 'reproduction_runs/round10_non_iid_generation'
SOURCE = ROOT / 'fuxian-round2/FedVaDE_source'
PAPER = ROOT / 'reproduction_runs/round4_source_audit/FedVaDE.pdf'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    if (OUT / 'source_audit.md').exists():
        raise SystemExit('Source audit exists; refusing overwrite')
    pages = [(p.extract_text() or '').lower() for p in PdfReader(PAPER).pages]
    if len(pages) != 6 or not any('sampling' in p for p in pages[1:4]):
        raise RuntimeError('Paper copy differs from expected six-page method text')
    model = (SOURCE / 'model.py').read_text()
    main_code = (SOURCE / 'main.py').read_text()
    required = ['GaussianMixture(n_components=K_l', 'K_l = len(np.unique(local_y))',
                'def gmm_sampling(', 'class Encoder', 'class Decoder', 'ELBO_Loss_server']
    if any(piece not in model for piece in required) or 'pre_epoch=50' not in main_code:
        raise RuntimeError('Author code differs from audited anchors')
    lines = [
        '# FedVaDE source and paper method audit', '',
        'Baseline: user-confirmed author-delivered FedVaDE_source. Author files are read only.',
        f'Paper PDF SHA256: `{sha(PAPER)}`; six pages, no full Appendix in this copy.',
        f'`main.py` SHA256: `{sha(SOURCE / "main.py")}`.',
        f'`model.py` SHA256: `{sha(SOURCE / "model.py")}`.',
        f'`dataloader.py` SHA256: `{sha(SOURCE / "dataloader.py")}`.', '',
        '- Encoder/decoder: `model.py` Encoder and Decoder use fully connected widths 500/500/2000 and a latent Gaussian of width 10 in the executed baselines. Decoder returns continuous values without a tabular validity projection. Paper p.3 describes the VAE latent representation.',
        '- Global prior: `VaDE` holds `rho_`, `mu_c`, `log_sigma2_c` for a diagonal Gaussian mixture. The GMM posterior is used for clustering assignment in `predict`; this is distinct from generation.',
        '- Author pretraining: `main.py` calls `pre_train(..., pre_epoch=50)`; `model.py` fits client diagonal GMMs to encoder means. It uses **true local labels to set K_l**, then fits a global GMM on latent samples. This is why the new CTG experiment is labeled `label_blind_fedvade_baseline`.',
        '- Server synthetic training: `main.py` calls `gmm_sampling` on the **client GMM states**, weighted by client record count. It decodes those latent samples with the server and respective client decoders, then optimizes `ELBO_Loss_server`. Paper pp.2–4 describes parameter exchange, latent sampling, decoding and server alignment. The server global GMM prior sampled for this diagnostic is not identical to the training-time mixture of client priors.',
        '- Client/server state: encoder/decoder parameters are weighted-averaged, server VAE is updated with generated samples, and the author code distributes the **pre-server-update** averaged VAE state to clients. GMM parameters are not part of that VAE average.',
        '- Generation vs assignment: (1) draw z from the chosen diagonal GMM, (2) decode z into x, (3) separately compute GMM posterior/argmax for a real encoded x. A generated x has no known true NSP label. GMM-component-conditioned generation is not true-class-conditioned generation.',
        '- No independent tabular generation-quality evaluation was found in the author entry code. The existing `validation` function reports clustering metrics. Paper p.5 discusses synthetic-sample count M but does not provide a held-out rare-mode generation protocol.',
        '- CTG adaptation: replace only the data loader with a fixed preprocessed partition and local K selection with unlabeled BIC over 1–3 components. Keep the author VAE, ELBO, client training, server resampling/aggregation, schedule and parameter distribution. Fixed global K=3 is a preregistered model capacity, not computed from run labels.',
        '- The author Breast Cancer CSV has a sample ID as one of 31 inputs, full-data standardization, overlapping client records, and label-assisted local K. Round 10 Breast Cancer generation results are explicitly `author_protocol_diagnostic`, not held-out test evidence.',
        '',
    ]
    (OUT / 'source_audit.md').write_text('\n'.join(lines))
    print('SOURCE_AUDIT_OK', flush=True)


if __name__ == '__main__':
    main()
