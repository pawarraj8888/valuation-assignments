/* Simplified CDO model in plain JavaScript.
 *
 * The same steps as the Python package (python/cdo) and the Excel workbook:
 *   independent normals -> Cholesky -> u = N(x) -> t = ln(1 - u) / ln(1 - pi) -> default quarter
 *   -> BIS bond cash flows -> pool -> waterfall (Class A, Class B, equity).
 * python/tests/test_site_model.py runs this file in node and compares it with Python.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.CDOModel = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var SHORTFALL_TOLERANCE = 1e-9;

  // Standard normal CDF in double precision (Hart's rational approximation, as given by West 2005).
  function normCdf(x) {
    var a = Math.abs(x);
    var result;
    if (a > 37) {
      result = 0;
    } else {
      var e = Math.exp(-a * a / 2);
      if (a < 7.07106781186547) {
        var num = 3.52624965998911e-2 * a + 0.700383064443688;
        num = num * a + 6.37396220353165;
        num = num * a + 33.912866078383;
        num = num * a + 112.079291497871;
        num = num * a + 221.213596169931;
        num = num * a + 220.206867912376;
        var den = 8.83883476483184e-2 * a + 1.75566716318264;
        den = den * a + 16.064177579207;
        den = den * a + 86.7807322029461;
        den = den * a + 296.564248779674;
        den = den * a + 637.333633378831;
        den = den * a + 793.826512519948;
        den = den * a + 440.413735824752;
        result = e * num / den;
      } else {
        var b = a + 0.65;
        b = a + 4 / b;
        b = a + 3 / b;
        b = a + 2 / b;
        b = a + 1 / b;
        result = e / b / 2.506628274631;
      }
    }
    return x > 0 ? 1 - result : result;
  }

  // Inverse of normCdf by bisection: slow but simple, and only used a handful of times.
  function normInv(p) {
    var lo = -12, hi = 12;
    for (var i = 0; i < 100; i++) {
      var mid = (lo + hi) / 2;
      if (normCdf(mid) < p) { lo = mid; } else { hi = mid; }
    }
    return (lo + hi) / 2;
  }

  // Lower Cholesky factor L of a symmetric positive definite matrix (matrix = L L').
  function choleskyOf(matrix) {
    var n = matrix.length, L = [];
    for (var i = 0; i < n; i++) {
      L.push(new Array(n).fill(0));
      for (var j = 0; j <= i; j++) {
        var s = 0;
        for (var k = 0; k < j; k++) { s += L[i][k] * L[j][k]; }
        L[i][j] = i === j ? Math.sqrt(matrix[i][i] - s) : (matrix[i][j] - s) / L[j][j];
      }
    }
    return L;
  }

  // Lower Cholesky factor of the n x n matrix with 1 on the diagonal and rho elsewhere.
  function cholesky(n, rho) {
    var matrix = [];
    for (var i = 0; i < n; i++) {
      matrix.push(new Array(n).fill(rho));
      matrix[i][i] = 1;
    }
    return choleskyOf(matrix);
  }

  // Moment matching as shown in class: de-mean the draws, then multiply them by the inverse of the Cholesky
  // factor of their covariance. The result has column means of 0 and an identity covariance matrix.
  function momentMatch(Z) {
    var n = Z.length, m = Z[0].length, c, i, j, k;
    if (n <= m) { throw new Error("moment matching needs more cases than bonds"); }
    var mean = new Array(m).fill(0);
    for (c = 0; c < n; c++) { for (j = 0; j < m; j++) { mean[j] += Z[c][j]; } }
    for (j = 0; j < m; j++) { mean[j] /= n; }
    var D = Z.map(function (row) { return row.map(function (v, col) { return v - mean[col]; }); });
    var cov = [];
    for (i = 0; i < m; i++) {
      cov.push(new Array(m).fill(0));
      for (j = 0; j <= i; j++) {
        var s = 0;
        for (c = 0; c < n; c++) { s += D[c][i] * D[c][j]; }
        cov[i][j] = s / n;
        cov[j][i] = cov[i][j];
      }
    }
    var L = choleskyOf(cov);
    return D.map(function (d) {            // each case: solve L y = d
      var y = new Array(m);
      for (i = 0; i < m; i++) {
        var t = d[i];
        for (k = 0; k < i; k++) { t -= L[i][k] * y[k]; }
        y[i] = t / L[i][i];
      }
      return y;
    });
  }

  // Coupon every period and the face value with the last coupon.
  function promisedCashFlows(face, coupon, nPeriods, freq) {
    var cf = new Array(nPeriods).fill(face * coupon / freq);
    cf[nPeriods - 1] += face;
    return cf;
  }

  function validateDeal(p) {
    if (!(p.pd >= 0 && p.pd < 1)) { throw new Error("annual default probability must be in [0, 1)"); }
    if (!(p.lgd >= 0 && p.lgd <= 1)) { throw new Error("LGD must be in [0, 1]"); }
    if (!(p.rho >= 0 && p.rho < 1)) { throw new Error("correlation must be in [0, 1)"); }
    ["face", "coupon", "a_notional", "a_coupon", "b_notional", "b_coupon"].forEach(function (key) {
      if (!(p[key] >= 0)) { throw new Error(key + " cannot be negative"); }
    });
  }

  // Correlated normal, uniform, default time (years) and default quarter of every bond in one case.
  function caseDefaults(zRow, p, L) {
    var nPeriods = p.years * p.freq;
    var logSurvival = Math.log(1 - p.pd);
    var x = [], u = [], t = [], q = [];
    for (var i = 0; i < zRow.length; i++) {
      var xi = 0;
      for (var k = 0; k <= i; k++) { xi += L[i][k] * zRow[k]; }
      var ui = normCdf(xi);
      var ti = p.pd === 0 ? Infinity : Math.log(1 - ui) / logSurvival;
      var qi = Math.ceil(Math.min(ti * p.freq, nPeriods + 1));
      x.push(xi); u.push(ui); t.push(ti); q.push(Math.max(qi, 1));
    }
    return { x: x, u: u, t: t, q: q };
  }

  // The whole model for one deal on the independent normals Z (array of cases, each an array of bonds).
  function simulate(Z, p) {
    validateDeal(p);
    if (Z[0].length !== p.n_bonds) { throw new Error("random numbers do not match the number of bonds"); }
    var nPeriods = p.years * p.freq;
    var bond = promisedCashFlows(p.face, p.coupon, nPeriods, p.freq);
    var aDue = promisedCashFlows(p.a_notional, p.a_coupon, nPeriods, p.freq);
    var bDue = promisedCashFlows(p.b_notional, p.b_coupon, nPeriods, p.freq);
    var L = cholesky(p.n_bonds, p.rho);
    var out = { nPeriods: nPeriods, bondPromised: bond, poolPromised: bond.map(function (v) { return v * p.n_bonds; }),
                aDue: aDue, bDue: bDue, Q: [], nDefaults: [], pool: [], a: [], b: [], equity: [] };
    for (var c = 0; c < Z.length; c++) {
      var q = caseDefaults(Z[c], p, L).q;
      var pool = new Array(nPeriods), a = new Array(nPeriods), b = new Array(nPeriods), eq = new Array(nPeriods);
      for (var k = 0; k < nPeriods; k++) {
        var cash = 0;
        for (var i = 0; i < q.length; i++) { cash += bond[k] * (k + 1 >= q[i] ? 1 - p.lgd : 1); }
        pool[k] = cash;
        a[k] = Math.min(cash, aDue[k]);
        b[k] = Math.min(cash - a[k], bDue[k]);
        eq[k] = cash - a[k] - b[k];
      }
      out.Q.push(q);
      out.nDefaults.push(q.filter(function (v) { return v <= nPeriods; }).length);
      out.pool.push(pool); out.a.push(a); out.b.push(b); out.equity.push(eq);
    }
    return out;
  }

  function sum(values) {
    var s = 0;
    for (var i = 0; i < values.length; i++) { s += values[i]; }
    return s;
  }

  function mean(values) { return sum(values) / values.length; }

  // Sample standard deviation (n - 1), as numpy std(ddof=1).
  function std(values) {
    var m = mean(values), s = 0;
    for (var i = 0; i < values.length; i++) { s += (values[i] - m) * (values[i] - m); }
    return Math.sqrt(s / (values.length - 1));
  }

  // Percentile with linear interpolation, as numpy.percentile and Excel PERCENTILE.
  function percentile(values, pct) {
    var sorted = values.slice().sort(function (x, y) { return x - y; });
    var rank = pct / 100 * (sorted.length - 1);
    var lo = Math.floor(rank), hi = Math.ceil(rank);
    return sorted[lo] + (rank - lo) * (sorted[hi] - sorted[lo]);
  }

  function totals(rows) { return rows.map(sum); }

  function describe(values, promised) {
    return { promised: promised, mean: mean(values), std: std(values), se: std(values) / Math.sqrt(values.length), min: Math.min.apply(null, values),
             p5: percentile(values, 5), median: percentile(values, 50), p95: percentile(values, 95),
             max: Math.max.apply(null, values), meanOverPromised: promised ? mean(values) / promised : NaN };
  }

  function shortfalls(paidRows, due) {
    var dueTotal = sum(due);
    return paidRows.map(function (row) { return dueTotal - sum(row) > SHORTFALL_TOLERANCE ? 1 : 0; });
  }

  // Key results for one deal: the same quantities as summary_row in python/cdo/analysis.py.
  function summaryRow(Z, p) {
    var r = simulate(Z, p);
    var pool = totals(r.pool), equity = totals(r.equity), bPaid = totals(r.b);
    var bDueTotal = sum(r.bDue);
    return {
      "avg defaults": mean(r.nDefaults),
      "P(no default)": mean(r.nDefaults.map(function (n) { return n === 0 ? 1 : 0; })),
      "P(4+ defaults)": mean(r.nDefaults.map(function (n) { return n >= 4 ? 1 : 0; })),
      "pool mean": mean(pool),
      "pool 5th pct": percentile(pool, 5),
      "equity mean": mean(equity),
      "equity std": std(equity),
      "equity 5th pct": percentile(equity, 5),
      "equity min": Math.min.apply(null, equity),
      "P(A shortfall)": mean(shortfalls(r.a, r.aDue)),
      "P(B shortfall)": mean(shortfalls(r.b, r.bDue)),
      "B paid / due": bDueTotal ? mean(bPaid) / bDueTotal : NaN
    };
  }

  function binomialPmf(k, n, p) {
    var c = 1;
    for (var i = 0; i < k; i++) { c = c * (n - i) / (i + 1); }
    return c * Math.pow(p, k) * Math.pow(1 - p, n - k);
  }

  // Exact distribution of the number of defaults: equal correlation = one common factor M, and given M
  // the bonds default independently. The binomial is averaged over M with Simpson's rule.
  function exactDefaultCountDistribution(nBonds, pDefault, rho) {
    var out = new Array(nBonds + 1).fill(0), k;
    if (rho === 0 || pDefault === 0 || pDefault === 1) {
      for (k = 0; k <= nBonds; k++) { out[k] = binomialPmf(k, nBonds, pDefault); }
      return out;
    }
    var threshold = normInv(pDefault), steps = 2000, lo = -9, h = 18 / steps;
    for (var s = 0; s <= steps; s++) {
      var m = lo + s * h;
      var weight = (s === 0 || s === steps ? 1 : s % 2 ? 4 : 2) * h / 3 * Math.exp(-m * m / 2) / 2.506628274631;
      var conditional = normCdf((threshold - Math.sqrt(rho) * m) / Math.sqrt(1 - rho));
      for (k = 0; k <= nBonds; k++) { out[k] += weight * binomialPmf(k, nBonds, conditional); }
    }
    return out;
  }

  // Exact expected pool cash flow per period: promised x [(1 - LGD) + LGD x (1 - pi)^(k / freq)].
  function expectedPoolCashFlows(p) {
    var nPeriods = p.years * p.freq;
    return promisedCashFlows(p.face, p.coupon, nPeriods, p.freq).map(function (v, k) {
      return p.n_bonds * v * ((1 - p.lgd) + p.lgd * Math.pow(1 - p.pd, (k + 1) / p.freq));
    });
  }

  return {
    normCdf: normCdf, normInv: normInv, cholesky: cholesky, choleskyOf: choleskyOf, momentMatch: momentMatch, promisedCashFlows: promisedCashFlows,
    validateDeal: validateDeal, caseDefaults: caseDefaults, simulate: simulate, sum: sum, mean: mean, std: std,
    percentile: percentile, totals: totals, describe: describe, shortfalls: shortfalls, summaryRow: summaryRow,
    binomialPmf: binomialPmf, exactDefaultCountDistribution: exactDefaultCountDistribution,
    expectedPoolCashFlows: expectedPoolCashFlows
  };
});
