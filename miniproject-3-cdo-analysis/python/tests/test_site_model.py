"""The project page recomputes the model in JavaScript (docs/model.js). Run it in node and compare."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdo import BASE_DEAL, exact_default_count_distribution, simulate, summary_row  # noqa: E402
from cdo.analysis import expected_pool_cash_flows  # noqa: E402

MODEL_JS = ROOT.parent / "docs" / "model.js"
STRESSED = {**BASE_DEAL, "pd": 0.12, "lgd": 1.0, "rho": 0.6, "b_notional": 30.0}
EXTREME = {**BASE_DEAL, "pd": 0.20, "lgd": 0.0, "rho": 0.95, "a_notional": 0.0, "b_notional": 60.0}
DEALS = [(0, BASE_DEAL), (1, STRESSED), (2, EXTREME)]
NORMAL_GRID = [x / 8.0 for x in range(-80, 81)]

SCRIPT = """
const model = require(process.argv[process.argv.length - 1]);
const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const out = { cdf: input.grid.map(model.normCdf), matched: model.momentMatch(input.raw), runs: [] };
for (const deal of input.deals) {
  const r = model.simulate(input.z, deal);
  out.runs.push({ Q: r.Q, pool: r.pool, a: r.a, b: r.b, equity: r.equity, summary: model.summaryRow(input.z, deal),
                  expected: model.expectedPoolCashFlows(deal),
                  exact: model.exactDefaultCountDistribution(deal.n_bonds, 1 - Math.pow(1 - deal.pd, deal.years), deal.rho) });
}
process.stdout.write(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def javascript(raw_normals, normals) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    payload = json.dumps({"raw": raw_normals.tolist(), "z": normals.tolist(), "deals": [deal for _, deal in DEALS], "grid": NORMAL_GRID})
    done = subprocess.run([node, "-e", SCRIPT, str(MODEL_JS)], input=payload, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_moment_matching_matches_python(javascript, normals):
    matched = np.array(javascript["matched"])
    assert np.allclose(matched, normals, rtol=0, atol=1e-12)
    # the page applies its own moment matching to the stored draws: it must lead to the same defaults
    assert np.array_equal(simulate(matched, BASE_DEAL)["Q"], simulate(normals, BASE_DEAL)["Q"])


def test_normal_cdf_matches_scipy(javascript):
    assert np.allclose(javascript["cdf"], norm.cdf(NORMAL_GRID), rtol=0, atol=1e-15)


@pytest.mark.parametrize("index, deal", DEALS)
def test_cash_flows_match_python_in_every_case(javascript, normals, index, deal):
    run = javascript["runs"][index]
    r = simulate(normals, deal)
    assert np.array_equal(np.array(run["Q"]), r["Q"])
    assert np.allclose(run["pool"], r["pool_cf"], rtol=0, atol=1e-9)
    assert np.allclose(run["a"], r["a_cf"], rtol=0, atol=1e-9)
    assert np.allclose(run["b"], r["b_cf"], rtol=0, atol=1e-9)
    assert np.allclose(run["equity"], r["eq_cf"], rtol=0, atol=1e-9)


@pytest.mark.parametrize("index, deal", DEALS)
def test_summary_statistics_match_python(javascript, normals, index, deal):
    expected = summary_row(normals, deal)
    summary = javascript["runs"][index]["summary"]
    assert set(summary) == set(expected)
    for key, value in expected.items():
        assert summary[key] == pytest.approx(value, abs=1e-9), key


@pytest.mark.parametrize("index, deal", DEALS)
def test_exact_benchmarks_match_python(javascript, index, deal):
    run = javascript["runs"][index]
    p_default = 1 - (1 - deal["pd"]) ** deal["years"]
    assert np.allclose(run["expected"], expected_pool_cash_flows(deal), rtol=0, atol=1e-9)
    assert np.allclose(run["exact"], exact_default_count_distribution(deal["n_bonds"], p_default, deal["rho"]), atol=1e-8)
