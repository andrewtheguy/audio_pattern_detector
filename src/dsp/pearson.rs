// ── Pearson Correlation ─────────────────────────────────────────────

/// Compute Pearson correlation coefficient between two equal-length f32 slices.
///
/// Returns *r* in \[-1, 1\].  Returns 0.0 if either series has zero variance
/// (e.g. constant arrays) or if the slices are empty.
pub fn pearson_correlation_1d(x: &[f32], y: &[f32]) -> f64 {
    assert_eq!(x.len(), y.len(), "slices must have the same length");
    let n = x.len() as f64;
    if n == 0.0 {
        return 0.0;
    }

    let mean_x = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let mean_y = y.iter().map(|&v| v as f64).sum::<f64>() / n;

    let (mut cov, mut var_x, mut var_y) = (0.0, 0.0, 0.0);
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi as f64 - mean_x;
        let dy = yi as f64 - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom == 0.0 {
        return 0.0;
    }
    cov / denom
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── pearson_correlation_1d ──────────────────────────────────────

    #[test]
    fn test_pearson_identical() {
        let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let r = pearson_correlation_1d(&a, &a);
        assert!(
            (r - 1.0).abs() < 1e-12,
            "identical arrays should give r=1.0, got {r}"
        );
    }

    #[test]
    fn test_pearson_negated() {
        let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let b: Vec<f32> = a.iter().map(|&v| -v).collect();
        let r = pearson_correlation_1d(&a, &b);
        assert!(
            (r - (-1.0)).abs() < 1e-12,
            "negated arrays should give r=-1.0, got {r}"
        );
    }

    #[test]
    fn test_pearson_constant_returns_zero() {
        let a = [3.0_f32; 5];
        let b = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let r = pearson_correlation_1d(&a, &b);
        assert!(
            (r).abs() < 1e-12,
            "constant array should give r=0.0, got {r}"
        );
    }

    #[test]
    fn test_pearson_empty() {
        let r = pearson_correlation_1d(&[], &[]);
        assert!((r).abs() < 1e-12, "empty arrays should give r=0.0, got {r}");
    }

    #[test]
    fn test_pearson_known_value() {
        // x=[1,2,3], y=[2,4,6] → perfectly correlated
        let x = [1.0_f32, 2.0, 3.0];
        let y = [2.0_f32, 4.0, 6.0];
        let r = pearson_correlation_1d(&x, &y);
        assert!(
            (r - 1.0).abs() < 1e-12,
            "linearly scaled should give r=1.0, got {r}"
        );
    }

    #[test]
    fn test_pearson_scaled_and_shifted() {
        // r is invariant to linear transforms: y = 3x + 10
        let x = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let y: Vec<f32> = x.iter().map(|&v| 3.0 * v + 10.0).collect();
        let r = pearson_correlation_1d(&x, &y);
        assert!(
            (r - 1.0).abs() < 1e-12,
            "affine transform should give r=1.0, got {r}"
        );
    }

    #[test]
    #[should_panic(expected = "slices must have the same length")]
    fn test_pearson_mismatched_lengths() {
        pearson_correlation_1d(&[1.0, 2.0], &[1.0]);
    }
}
