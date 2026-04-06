use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

#[cfg(feature = "python")]
mod python;

// ── BS.1770 Loudness ─────────────────────────────────────────────────

/// Compute K-weighting biquad filter coefficients for a given sample rate.
///
/// Returns `(b_shelf, a_shelf, b_hpass, a_hpass)` — two sets of biquad
/// coefficients per ITU-R BS.1770 (high-shelf at 1500 Hz, high-pass at 38 Hz).
fn k_weighting_coefficients(rate: f64) -> ([f64; 3], [f64; 3], [f64; 3], [f64; 3]) {
    // ── High shelf: G=4dB, Q=1/√2, fc=1500Hz ──
    let g = 4.0_f64;
    let q = std::f64::consts::FRAC_1_SQRT_2;
    let fc = 1500.0;

    let a_val = 10.0_f64.powf(g / 40.0);
    let w0 = 2.0 * std::f64::consts::PI * (fc / rate);
    let alpha = w0.sin() / (2.0 * q);
    let cos_w0 = w0.cos();
    let two_sqrt_a_alpha = 2.0 * a_val.sqrt() * alpha;

    let b0 = a_val * ((a_val + 1.0) + (a_val - 1.0) * cos_w0 + two_sqrt_a_alpha);
    let b1 = -2.0 * a_val * ((a_val - 1.0) + (a_val + 1.0) * cos_w0);
    let b2 = a_val * ((a_val + 1.0) + (a_val - 1.0) * cos_w0 - two_sqrt_a_alpha);
    let a0s = (a_val + 1.0) - (a_val - 1.0) * cos_w0 + two_sqrt_a_alpha;
    let a1s = 2.0 * ((a_val - 1.0) - (a_val + 1.0) * cos_w0);
    let a2s = (a_val + 1.0) - (a_val - 1.0) * cos_w0 - two_sqrt_a_alpha;

    let b_shelf = [b0 / a0s, b1 / a0s, b2 / a0s];
    let a_shelf = [1.0, a1s / a0s, a2s / a0s];

    // ── High pass: Q=0.5, fc=38Hz ──
    let q2 = 0.5;
    let fc2 = 38.0;
    let w0_2 = 2.0 * std::f64::consts::PI * (fc2 / rate);
    let alpha2 = w0_2.sin() / (2.0 * q2);
    let cos_w0_2 = w0_2.cos();

    let hb0 = (1.0 + cos_w0_2) / 2.0;
    let hb1 = -(1.0 + cos_w0_2);
    let hb2 = (1.0 + cos_w0_2) / 2.0;
    let ha0 = 1.0 + alpha2;
    let ha1 = -2.0 * cos_w0_2;
    let ha2 = 1.0 - alpha2;

    let b_hpass = [hb0 / ha0, hb1 / ha0, hb2 / ha0];
    let a_hpass = [1.0, ha1 / ha0, ha2 / ha0];

    (b_shelf, a_shelf, b_hpass, a_hpass)
}

/// Direct-form II transposed IIR filter (biquad), equivalent to
/// `scipy.signal.lfilter(b, a, data)` for second-order sections.
#[cfg_attr(not(test), allow(dead_code))]
fn lfilter_biquad(b: &[f64; 3], a: &[f64; 3], data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![0.0_f64; n];
    let mut d1 = 0.0_f64;
    let mut d2 = 0.0_f64;

    for i in 0..n {
        let x = data[i];
        let y = b[0] * x + d1;
        d1 = b[1] * x - a[1] * y + d2;
        d2 = b[2] * x - a[2] * y;
        out[i] = y;
    }
    out
}

#[inline]
fn biquad_step(b: &[f64; 3], a: &[f64; 3], d1: &mut f64, d2: &mut f64, x: f64) -> f64 {
    let y = b[0] * x + *d1;
    *d1 = b[1] * x - a[1] * y + *d2;
    *d2 = b[2] * x - a[2] * y;
    y
}

/// Apply the two BS.1770 K-weighting filters in a single pass and return
/// a prefix sum of squared output energy.
fn k_weighted_squared_prefix(
    data: &[f32],
    b_shelf: &[f64; 3],
    a_shelf: &[f64; 3],
    b_hpass: &[f64; 3],
    a_hpass: &[f64; 3],
) -> Vec<f64> {
    let mut prefix = vec![0.0_f64; data.len() + 1];
    let mut shelf_d1 = 0.0_f64;
    let mut shelf_d2 = 0.0_f64;
    let mut hpass_d1 = 0.0_f64;
    let mut hpass_d2 = 0.0_f64;

    for (idx, &sample) in data.iter().enumerate() {
        let shelf_out = biquad_step(
            b_shelf,
            a_shelf,
            &mut shelf_d1,
            &mut shelf_d2,
            sample as f64,
        );
        let filtered = biquad_step(b_hpass, a_hpass, &mut hpass_d1, &mut hpass_d2, shelf_out);
        prefix[idx + 1] = prefix[idx] + filtered * filtered;
    }

    prefix
}

#[inline]
fn loudness_block_bounds(
    block_index: usize,
    window_samples: f64,
    hop_samples: f64,
    signal_len: usize,
) -> (usize, usize) {
    let start = (block_index as f64 * hop_samples) as usize;
    let end = (block_index as f64 * hop_samples + window_samples) as usize;
    (start, end.min(signal_len))
}

/// Measure integrated gated loudness per ITU-R BS.1770-4.
///
/// Input must be mono f32 samples in [-1, 1].
/// Returns loudness in dB LUFS (may be `-inf` for silence).
pub fn integrated_loudness(data: &[f32], sample_rate: u32, block_size: f64) -> f64 {
    const LUFS_OFFSET: f64 = -0.691;
    const ABSOLUTE_GATE: f64 = -70.0;
    const OVERLAP: f64 = 0.75;

    let rate = sample_rate as f64;
    let n = data.len();
    if n == 0 {
        return f64::NEG_INFINITY;
    }

    let (b_shelf, a_shelf, b_hpass, a_hpass) = k_weighting_coefficients(rate);
    let squared_prefix = k_weighted_squared_prefix(data, &b_shelf, &a_shelf, &b_hpass, &a_hpass);

    // Gating parameters.
    let t_g = block_size; // default 0.4s
    let step = 1.0 - OVERLAP;
    let window_samples = t_g * rate;
    let hop_samples = window_samples * step;

    let t = n as f64 / rate;
    let num_blocks = ((t - t_g) / (t_g * step)).round() as i64 + 1;
    if num_blocks <= 0 {
        // Signal shorter than one block — compute mean square directly.
        let ms = squared_prefix[n] / n as f64;
        if ms <= 0.0 {
            return f64::NEG_INFINITY;
        }
        return LUFS_OFFSET + 10.0 * ms.log10();
    }
    let num_blocks = num_blocks as usize;

    // Absolute gating pass.
    let mut z_abs_sum = 0.0_f64;
    let mut z_abs_count = 0_usize;
    for j in 0..num_blocks {
        let (l, u) = loudness_block_bounds(j, window_samples, hop_samples, n);
        if l >= u {
            continue;
        }
        let ms = (squared_prefix[u] - squared_prefix[l]) / (u - l) as f64;
        if ms <= 0.0 {
            continue;
        }

        let loudness = LUFS_OFFSET + 10.0 * ms.log10();
        if loudness >= ABSOLUTE_GATE {
            z_abs_sum += ms;
            z_abs_count += 1;
        }
    }

    if z_abs_count == 0 {
        return f64::NEG_INFINITY;
    }

    // Average of gated blocks for relative threshold.
    let z_avg = z_abs_sum / z_abs_count as f64;
    let gamma_r = LUFS_OFFSET + 10.0 * z_avg.log10() - 10.0;

    // Relative gating pass.
    let mut z_rel_sum = 0.0_f64;
    let mut z_rel_count = 0_usize;
    for j in 0..num_blocks {
        let (l, u) = loudness_block_bounds(j, window_samples, hop_samples, n);
        if l >= u {
            continue;
        }
        let ms = (squared_prefix[u] - squared_prefix[l]) / (u - l) as f64;
        if ms <= 0.0 {
            continue;
        }

        let loudness = LUFS_OFFSET + 10.0 * ms.log10();
        if loudness > gamma_r && loudness >= ABSOLUTE_GATE {
            z_rel_sum += ms;
            z_rel_count += 1;
        }
    }

    if z_rel_count == 0 {
        return f64::NEG_INFINITY;
    }

    let z_avg_final = z_rel_sum / z_rel_count as f64;
    LUFS_OFFSET + 10.0 * z_avg_final.log10()
}

/// Normalize audio to a target loudness in dB LUFS with hard clipping.
///
/// Applies the gain needed to shift from `current_lufs` to `target_lufs`,
/// then hard-clips the output to [-1.0, 1.0].
pub fn loudness_normalize(data: &[f32], current_lufs: f64, target_lufs: f64) -> Vec<f32> {
    let delta = target_lufs - current_lufs;
    let gain = 10.0_f64.powf(delta / 20.0);

    data.iter()
        .map(|&x| ((x as f64) * gain).clamp(-1.0, 1.0) as f32)
        .collect()
}

// ── Resampling ───────────────────────────────────────────────────────

/// FFT-based resampling of a 1-D signal, matching `scipy.signal.resample`.
///
/// Uses the full complex FFT to truncate or zero-pad the frequency-domain
/// representation, exactly replicating scipy's spectrum manipulation.
pub fn resample_1d(data: &[f32], target_len: usize) -> Vec<f32> {
    let n = data.len();
    if n == 0 || target_len == 0 {
        return vec![0.0; target_len];
    }
    if n == target_len {
        return data.to_vec();
    }

    let m = target_len;

    // Forward complex FFT.
    let mut planner = FftPlanner::<f64>::new();
    let fft_fwd = planner.plan_fft_forward(n);
    let mut spectrum: Vec<Complex<f64>> =
        data.iter().map(|&v| Complex::new(v as f64, 0.0)).collect();
    fft_fwd.process(&mut spectrum);

    // Build new spectrum of length m, matching scipy's slice logic:
    //   N = min(num, Nx)
    //   Y[0:(N+1)//2]    = X[0:(N+1)//2]     (positive frequencies)
    //   Y[-(N-1)//2:]    = X[-(N-1)//2:]      (negative frequencies)
    let n_common = n.min(m);
    let pos = (n_common + 1) / 2; // number of positive-frequency bins to copy
    let neg = (n_common - 1) / 2; // number of negative-frequency bins to copy

    let mut new_spectrum = vec![Complex::new(0.0, 0.0); m];
    new_spectrum[..pos].copy_from_slice(&spectrum[..pos]);
    if neg > 0 {
        new_spectrum[m - neg..].copy_from_slice(&spectrum[n - neg..]);
    }

    // Inverse complex FFT.
    let fft_inv = planner.plan_fft_inverse(m);
    fft_inv.process(&mut new_spectrum);

    // Scale: rustfft inverse is un-normalised (factor m), and scipy applies
    // target/source.  Combined scale: target / (source * target) = 1/source.
    let scale = 1.0 / n as f64;
    new_spectrum.iter().map(|c| (c.re * scale) as f32).collect()
}

/// Resample a 1-D signal to `target_len` by partitioning it into windows
/// and keeping the maximum sample from each window.
///
/// Works for both downsampling and upsampling.  Guarantees
/// `output.len() == target_len`.  When upsampling, windows that map to the
/// same source sample simply repeat it.
pub fn resample_preserve_maxima_1d(data: &[f32], target_len: usize) -> Vec<f32> {
    if target_len == 0 || data.is_empty() {
        return Vec::new();
    }

    let n_points = data.len();
    let step_size = n_points as f64 / target_len as f64;
    let mut downsampled = Vec::with_capacity(target_len);

    for i in 0..target_len {
        let mut start_index = (i as f64 * step_size) as usize;
        let mut end_index = ((i + 1) as f64 * step_size) as usize;

        // Guarantee at least one sample per window
        if end_index <= start_index {
            end_index = start_index + 1;
        }

        // Clamp into [0, n_points)
        if start_index >= n_points {
            start_index = n_points - 1;
        }
        if end_index > n_points {
            end_index = n_points;
        }

        let max_value = data[start_index..end_index]
            .iter()
            .copied()
            .reduce(f32::max)
            .expect("non-empty window must have a maximum");
        downsampled.push(max_value);
    }

    downsampled
}


// ── Simpson's rule ───────────────────────────────────────────────────

/// Composite Simpson's rule for uniformly spaced data with unit spacing (dx=1).
///
/// Matches `scipy.integrate.simpson(y)`: uses Simpson's 1/3 for even
/// intervals and the Cartwright correction for the trailing odd interval.
pub fn simpson_1d(y: &[f64]) -> f64 {
    let n = y.len();
    if n < 2 {
        return 0.0;
    }
    if n == 2 {
        return (y[0] + y[1]) / 2.0;
    }

    if n % 2 == 1 {
        // Odd points → even intervals: standard composite Simpson's 1/3.
        composite_simpson_13(y)
    } else {
        // Even points → odd intervals: Simpson's 1/3 on first N-3 intervals,
        // then Cartwright correction for the last interval.
        let base = composite_simpson_13(&y[..n - 1]); // odd slice, covers [0, n-2]
                                                      // Cartwright correction (h=1):
                                                      //   alpha = 5/12, beta = 8/12, eta = 1/12
        let correction =
            (5.0 / 12.0) * y[n - 1] + (8.0 / 12.0) * y[n - 2] - (1.0 / 12.0) * y[n - 3];
        base + correction
    }
}

/// Standard composite Simpson's 1/3 rule for an odd number of points.
fn composite_simpson_13(y: &[f64]) -> f64 {
    debug_assert!(y.len() >= 3 && y.len() % 2 == 1);
    let n = y.len();
    let mut sum = y[0] + y[n - 1];
    for i in (1..n - 1).step_by(2) {
        sum += 4.0 * y[i];
    }
    for i in (2..n - 1).step_by(2) {
        sum += 2.0 * y[i];
    }
    sum / 3.0
}

// ── Peak finding ─────────────────────────────────────────────────────

/// Options for peak finding.
pub struct FindPeaksOptions {
    pub height: Option<f32>,
    pub distance: Option<usize>,
    pub prominence: Option<f32>,
}

/// Find peaks (local maxima) in a 1-D signal.
///
/// Matches the semantics of `scipy.signal.find_peaks` for the supported
/// parameters: `height`, `distance`, and `prominence`.
///
/// Returns a sorted vector of peak indices.
pub fn find_peaks_1d(data: &[f32], options: &FindPeaksOptions) -> Vec<usize> {
    let mut peaks = local_maxima_1d(data);

    if let Some(min_height) = options.height {
        filter_by_height(data, &mut peaks, min_height);
    }

    if let Some(min_distance) = options.distance {
        filter_by_distance(data, &mut peaks, min_distance);
    }

    if let Some(min_prominence) = options.prominence {
        filter_by_prominence(data, &mut peaks, min_prominence);
    }

    peaks
}

/// Detect all local maxima in `data`.
///
/// A sample is a local maximum when it is strictly greater than both its
/// immediate neighbours.  For plateaus (runs of identical values that are
/// higher than the values on both sides) the midpoint index (rounded down)
/// is returned, matching scipy's behaviour.
fn local_maxima_1d(data: &[f32]) -> Vec<usize> {
    let n = data.len();
    if n < 3 {
        return vec![];
    }

    let mut peaks = Vec::new();
    let mut i = 1;
    while i < n - 1 {
        if data[i - 1] < data[i] {
            let left_edge = i;
            // Advance through equal values (plateau).
            while i + 1 < n && data[i] == data[i + 1] {
                i += 1;
            }
            let right_edge = i;
            // Confirm the right side drops.
            if i + 1 < n && data[i] > data[i + 1] {
                peaks.push((left_edge + right_edge) / 2);
            }
        }
        i += 1;
    }
    peaks
}

/// Keep only peaks whose value is at least `min_height`.
fn filter_by_height(data: &[f32], peaks: &mut Vec<usize>, min_height: f32) {
    peaks.retain(|&idx| data[idx] >= min_height);
}

/// Keep only the tallest peaks when multiple peaks fall within `min_distance`
/// of each other.  Matches scipy's greedy tallest-first strategy.
fn filter_by_distance(data: &[f32], peaks: &mut Vec<usize>, min_distance: usize) {
    if peaks.is_empty() || min_distance == 0 {
        return;
    }

    let n = peaks.len();

    // Build a priority order: tallest first, break ties by lower index.
    let mut priority: Vec<usize> = (0..n).collect();
    priority.sort_by(|&a, &b| {
        data[peaks[b]]
            .partial_cmp(&data[peaks[a]])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });

    let mut keep = vec![true; n];

    for &idx in &priority {
        if !keep[idx] {
            continue;
        }
        // Scan left in the peaks array.
        let mut j = idx;
        while j > 0 {
            j -= 1;
            if peaks[idx] - peaks[j] >= min_distance {
                break;
            }
            keep[j] = false;
        }
        // Scan right in the peaks array.
        for j in (idx + 1)..n {
            if peaks[j] - peaks[idx] >= min_distance {
                break;
            }
            keep[j] = false;
        }
    }

    let mut write = 0;
    for read in 0..n {
        if keep[read] {
            peaks[write] = peaks[read];
            write += 1;
        }
    }
    peaks.truncate(write);
}

/// Compute the prominence of a single peak.
///
/// Prominence is defined as:
///   `data[peak] - max(left_base, right_base)`
///
/// where `left_base` is the minimum value between the peak and the nearest
/// higher peak (or array boundary) to the left, and `right_base` likewise
/// to the right.
#[cfg_attr(not(test), allow(dead_code))]
fn compute_prominence(data: &[f32], peak_idx: usize) -> f32 {
    let peak_val = data[peak_idx];

    // Scan left: find the minimum between peak and the nearest higher value
    // (or the left boundary).
    let mut left_min = peak_val;
    for j in (0..peak_idx).rev() {
        if data[j] < left_min {
            left_min = data[j];
        }
        if data[j] > peak_val {
            break;
        }
    }

    // Scan right.
    let mut right_min = peak_val;
    for j in (peak_idx + 1)..data.len() {
        if data[j] < right_min {
            right_min = data[j];
        }
        if data[j] > peak_val {
            break;
        }
    }

    peak_val - left_min.max(right_min)
}

/// Segment tree for fast range-min queries over the input signal.
struct RangeMinTree {
    size: usize,
    values: Vec<f32>,
}

impl RangeMinTree {
    fn new(data: &[f32]) -> Self {
        let size = data.len().max(1).next_power_of_two();
        let mut values = vec![f32::INFINITY; size * 2];
        values[size..size + data.len()].copy_from_slice(data);

        for idx in (1..size).rev() {
            values[idx] = values[idx * 2].min(values[idx * 2 + 1]);
        }

        Self { size, values }
    }

    /// Return the minimum in the half-open interval `[start, end)`.
    fn min_in_range(&self, start: usize, end: usize) -> f32 {
        if start >= end {
            return f32::INFINITY;
        }

        let mut left = start + self.size;
        let mut right = end + self.size;
        let mut result = f32::INFINITY;

        while left < right {
            if left % 2 == 1 {
                result = result.min(self.values[left]);
                left += 1;
            }
            if right % 2 == 1 {
                right -= 1;
                result = result.min(self.values[right]);
            }
            left /= 2;
            right /= 2;
        }

        result
    }
}

/// Return the nearest strictly higher sample on each side of every index.
///
/// Equal-height samples are skipped so prominence matches the current
/// `compute_prominence` semantics, where only values `>` the peak stop the scan.
fn nearest_strictly_greater_indices(data: &[f32]) -> (Vec<Option<usize>>, Vec<Option<usize>>) {
    let n = data.len();
    let mut left = vec![None; n];
    let mut right = vec![None; n];
    let mut stack = Vec::with_capacity(n);

    for idx in 0..n {
        while let Some(&prev) = stack.last() {
            if data[prev] <= data[idx] {
                stack.pop();
            } else {
                break;
            }
        }
        left[idx] = stack.last().copied();
        stack.push(idx);
    }

    stack.clear();

    for idx in (0..n).rev() {
        while let Some(&next) = stack.last() {
            if data[next] <= data[idx] {
                stack.pop();
            } else {
                break;
            }
        }
        right[idx] = stack.last().copied();
        stack.push(idx);
    }

    (left, right)
}

fn compute_prominence_with_preprocessing(
    data: &[f32],
    peak_idx: usize,
    left_greater: &[Option<usize>],
    right_greater: &[Option<usize>],
    range_min: &RangeMinTree,
) -> f32 {
    let peak_val = data[peak_idx];

    let left_start = left_greater[peak_idx].map_or(0, |idx| idx + 1);
    let left_min = range_min.min_in_range(left_start, peak_idx).min(peak_val);

    let right_end = right_greater[peak_idx].unwrap_or(data.len());
    let right_min = range_min
        .min_in_range(peak_idx + 1, right_end)
        .min(peak_val);

    peak_val - left_min.max(right_min)
}

/// Keep only peaks whose prominence is at least `min_prominence`.
fn filter_by_prominence(data: &[f32], peaks: &mut Vec<usize>, min_prominence: f32) {
    if peaks.is_empty() {
        return;
    }

    let (left_greater, right_greater) = nearest_strictly_greater_indices(data);
    let range_min = RangeMinTree::new(data);

    peaks.retain(|&idx| {
        compute_prominence_with_preprocessing(data, idx, &left_greater, &right_greater, &range_min)
            >= min_prominence
    });
}

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

    // ── local_maxima_1d ──────────────────────────────────────────────

    #[test]
    fn test_local_maxima_simple() {
        let data = [0.0, 1.0, 0.0, 2.0, 0.0];
        assert_eq!(local_maxima_1d(&data), vec![1, 3]);
    }

    #[test]
    fn test_local_maxima_empty_and_short() {
        assert_eq!(local_maxima_1d(&[]), Vec::<usize>::new());
        assert_eq!(local_maxima_1d(&[1.0]), Vec::<usize>::new());
        assert_eq!(local_maxima_1d(&[1.0, 2.0]), Vec::<usize>::new());
    }

    #[test]
    fn test_local_maxima_plateau_even() {
        // Plateau [1,1] spans indices 1..2 → midpoint = 1
        let data = [0.0, 1.0, 1.0, 0.0];
        assert_eq!(local_maxima_1d(&data), vec![1]);
    }

    #[test]
    fn test_local_maxima_plateau_odd() {
        // Plateau [1,1,1] spans indices 1..3 → midpoint = 2
        let data = [0.0, 1.0, 1.0, 1.0, 0.0];
        assert_eq!(local_maxima_1d(&data), vec![2]);
    }

    #[test]
    fn test_local_maxima_monotonic() {
        let ascending = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(local_maxima_1d(&ascending), Vec::<usize>::new());

        let descending = [5.0, 4.0, 3.0, 2.0, 1.0];
        assert_eq!(local_maxima_1d(&descending), Vec::<usize>::new());
    }

    // ── height filter ────────────────────────────────────────────────

    #[test]
    fn test_height_filter() {
        let data = [0.0, 1.0, 0.0, 2.0, 0.0];
        let opts = FindPeaksOptions {
            height: Some(1.5),
            distance: None,
            prominence: None,
        };
        assert_eq!(find_peaks_1d(&data, &opts), vec![3]);
    }

    // ── distance filter ──────────────────────────────────────────────

    #[test]
    fn test_distance_keeps_tallest() {
        // Two peaks 2 apart, distance=3 → keep the tallest.
        let data = [0.0, 3.0, 0.0, 5.0, 0.0];
        let opts = FindPeaksOptions {
            height: None,
            distance: Some(3),
            prominence: None,
        };
        assert_eq!(find_peaks_1d(&data, &opts), vec![3]);
    }

    #[test]
    fn test_distance_allows_far_peaks() {
        let data = [0.0, 3.0, 0.0, 0.0, 0.0, 5.0, 0.0];
        let opts = FindPeaksOptions {
            height: None,
            distance: Some(3),
            prominence: None,
        };
        assert_eq!(find_peaks_1d(&data, &opts), vec![1, 5]);
    }

    // ── prominence filter ────────────────────────────────────────────

    #[test]
    fn test_prominence_simple() {
        // Peak at index 3 (value 5), troughs at 0 on both sides → prominence 5.
        let data = [0.0, 0.0, 0.0, 5.0, 0.0, 0.0];
        assert_eq!(compute_prominence(&data, 3), 5.0);
    }

    #[test]
    fn test_prominence_with_higher_neighbor() {
        // data = [0, 3, 1, 5, 0]
        // Peak at 1 (val 3): scan left → 0 (min 0), scan right → hits 5 > 3 at idx 3
        //   right_min = min(1, 5) but stops at 3: min of data[2]=1 before hitting data[3]=5
        //   Actually: right_min starts at 3.0, data[2]=1.0 < 3.0 → right_min=1.0,
        //   data[3]=5.0 > 3.0 → break.  left_min: data[0]=0.0.
        //   prominence = 3.0 - max(0.0, 1.0) = 2.0
        let data = [0.0, 3.0, 1.0, 5.0, 0.0];
        assert_eq!(compute_prominence(&data, 1), 2.0);
    }

    #[test]
    fn test_prominence_filter() {
        let data = [0.0, 1.0, 0.5, 2.0, 0.0];
        // Peak at 1: prominence = 1.0 - max(0.0, 0.5) = 0.5
        // Peak at 3: prominence = 2.0 - max(0.5, 0.0) = 1.5
        let opts = FindPeaksOptions {
            height: None,
            distance: None,
            prominence: Some(1.0),
        };
        assert_eq!(find_peaks_1d(&data, &opts), vec![3]);
    }

    #[test]
    fn test_prominence_ignores_equal_height_peaks() {
        let data = [0.0, 5.0, 0.0, 5.0, 0.0];
        let opts = FindPeaksOptions {
            height: None,
            distance: None,
            prominence: Some(4.0),
        };
        assert_eq!(find_peaks_1d(&data, &opts), vec![1, 3]);
    }

    // ── combined filters ─────────────────────────────────────────────

    #[test]
    fn test_combined_height_and_distance() {
        let data = [0.0, 2.0, 0.0, 1.0, 0.0, 3.0, 0.0];
        let opts = FindPeaksOptions {
            height: Some(1.5),
            distance: Some(3),
            prominence: None,
        };
        // After height filter: peaks at 1 (2.0) and 5 (3.0). Distance 4 ≥ 3 → both kept.
        assert_eq!(find_peaks_1d(&data, &opts), vec![1, 5]);
    }

    #[test]
    fn test_all_filters() {
        let data = [0.0, 5.0, 4.5, 5.0, 0.0, 3.0, 0.0];
        let opts = FindPeaksOptions {
            height: Some(2.0),
            distance: Some(2),
            prominence: Some(1.0),
        };
        // Local maxima: 1 (5.0), 3 (5.0), 5 (3.0)
        // Height ≥ 2.0: all pass
        // Distance ≥ 2: peaks at 1 and 3 are 2 apart (≥2 → both kept), 5 is 2 from 3 (≥2 → kept)
        // Prominence:
        //   Peak 1 (5.0): left_min=0.0, right scan: 4.5 then 5.0>5.0? no (equal) → continues to 0.0, 3.0, 0.0
        //     Actually scan right from 1: data[2]=4.5 < 5.0 (min=4.5), data[3]=5.0 = 5.0 (not >), data[4]=0.0 (min=0.0), data[5]=3.0 (not > 5.0), data[6]=0.0
        //     No value > 5.0 found → right_min = 0.0
        //     left_min: data[0]=0.0, no value > 5.0 → left_min = 0.0
        //     prominence = 5.0 - max(0.0, 0.0) = 5.0 ✓
        //   Peak 3 (5.0): same logic → prominence 5.0 ✓
        //   Peak 5 (3.0): left scan data[4]=0.0, data[3]=5.0>3.0 break → left_min=0.0
        //     right scan data[6]=0.0 → right_min=0.0
        //     prominence = 3.0 ✓
        // All pass
        assert_eq!(find_peaks_1d(&data, &opts), vec![1, 3, 5]);
    }

    #[test]
    fn test_no_peaks() {
        let data = [1.0, 1.0, 1.0, 1.0];
        let opts = FindPeaksOptions {
            height: None,
            distance: None,
            prominence: None,
        };
        assert_eq!(find_peaks_1d(&data, &opts), Vec::<usize>::new());
    }

    // ── resample ─────────────────────────────────────────────────────

    #[test]
    fn test_resample_identity() {
        let data = [1.0_f32, 2.0, 3.0, 4.0];
        let out = resample_1d(&data, 4);
        for (a, b) in out.iter().zip(data.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} != {b}");
        }
    }

    #[test]
    fn test_resample_empty() {
        assert_eq!(resample_1d(&[], 0), Vec::<f32>::new());
        assert_eq!(resample_1d(&[], 5), vec![0.0; 5]);
        assert_eq!(resample_1d(&[1.0, 2.0], 0), Vec::<f32>::new());
    }

    #[test]
    fn test_resample_downsample() {
        // Sine wave at 8 points → 4 points.
        let n = 8;
        let data: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * i as f32 / n as f32).sin())
            .collect();
        let out = resample_1d(&data, 4);
        assert_eq!(out.len(), 4);
        // A single-cycle sine resampled to 4 points should still be ~sinusoidal.
        // Values should be close to sin(0), sin(pi/2), sin(pi), sin(3pi/2) = 0, 1, 0, -1
        assert!(out[0].abs() < 0.1);
        assert!((out[1] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_resample_upsample() {
        let data = [0.0_f32, 1.0, 0.0];
        let out = resample_1d(&data, 6);
        assert_eq!(out.len(), 6);
    }

    #[test]
    fn test_resample_preserve_maxima_identity() {
        let data = [1.0_f32, 3.0, 2.0, 4.0];
        let out = resample_preserve_maxima_1d(&data, data.len());
        assert_eq!(out, data);
    }

    #[test]
    fn test_resample_preserve_maxima_window_maxes() {
        let data = [1.0_f32, 5.0, 2.0, 4.0, 3.0, 6.0];
        let out = resample_preserve_maxima_1d(&data, 3);
        assert_eq!(out, vec![5.0, 4.0, 6.0]);
    }

    #[test]
    fn test_resample_preserve_maxima_short_input() {
        // Upsampling: 3 samples → 5 windows (step_size = 0.6).
        // i=0: [0..1)→1.0, i=1: [0..1)→1.0, i=2: [1..2)→2.0,
        // i=3: [1..2)→2.0, i=4: [2..3)→3.0
        let data = [1.0_f32, 2.0, 3.0];
        let out = resample_preserve_maxima_1d(&data, 5);
        assert_eq!(out, vec![1.0, 1.0, 2.0, 2.0, 3.0]);
    }

    #[test]
    fn test_resample_preserve_maxima_upsample_single() {
        // Edge case: 1 sample → 4 windows should repeat the value.
        let data = [7.0_f32];
        let out = resample_preserve_maxima_1d(&data, 4);
        assert_eq!(out, vec![7.0, 7.0, 7.0, 7.0]);
    }

    #[test]
    fn test_resample_preserve_maxima_upsample_two_to_six() {
        // 2 samples → 6 windows (step_size = 0.333): each source sample
        // maps to 3 windows.
        let data = [1.0_f32, 5.0];
        let out = resample_preserve_maxima_1d(&data, 6);
        assert_eq!(out, vec![1.0, 1.0, 1.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_resample_preserve_maxima_upsample_preserves_all_values() {
        // 5 samples → 20 windows (step_size = 0.25): each source sample
        // maps to exactly 4 windows.
        let data = [3.0_f32, 1.0, 4.0, 1.0, 5.0];
        let out = resample_preserve_maxima_1d(&data, 20);
        assert_eq!(
            out,
            vec![
                3.0, 3.0, 3.0, 3.0,
                1.0, 1.0, 1.0, 1.0,
                4.0, 4.0, 4.0, 4.0,
                1.0, 1.0, 1.0, 1.0,
                5.0, 5.0, 5.0, 5.0,
            ]
        );
    }

    #[test]
    fn test_resample_preserve_maxima_same_length() {
        // target_len == data.len() should be identity.
        let data = [2.0_f32, 8.0, 3.0, 7.0, 1.0];
        let out = resample_preserve_maxima_1d(&data, 5);
        assert_eq!(out, data);
    }

    #[test]
    fn test_resample_preserve_maxima_empty_and_zero_target() {
        assert_eq!(resample_preserve_maxima_1d(&[], 4), Vec::<f32>::new());
        assert_eq!(
            resample_preserve_maxima_1d(&[1.0, 2.0], 0),
            Vec::<f32>::new()
        );
    }

    // ── lttb_1d ─────────────────────────────────────────────────────

    // ── simpson ──────────────────────────────────────────────────────

    #[test]
    fn test_simpson_constant() {
        // Integral of y=2 over [0,4] with unit spacing = 8.
        let y = [2.0_f64; 5]; // 5 points, 4 intervals
        let area = simpson_1d(&y);
        assert!((area - 8.0).abs() < 1e-10, "got {area}");
    }

    #[test]
    fn test_simpson_linear() {
        // Integral of y=x from 0 to 4 = 8.
        let y: Vec<f64> = (0..5).map(|i| i as f64).collect();
        let area = simpson_1d(&y);
        assert!((area - 8.0).abs() < 1e-10, "got {area}");
    }

    #[test]
    fn test_simpson_quadratic() {
        // Integral of y=x^2 from 0 to 4 = 64/3 ≈ 21.333...
        // Simpson's rule is exact for polynomials up to degree 3.
        let y: Vec<f64> = (0..5).map(|i| (i as f64).powi(2)).collect();
        let area = simpson_1d(&y);
        assert!((area - 64.0 / 3.0).abs() < 1e-10, "got {area}");
    }

    #[test]
    fn test_simpson_even_points() {
        // 4 points (even): Simpson 1/3 on [0,2] + Cartwright for [2,3].
        // y=x^2 at 0,1,2,3 → [0, 1, 4, 9]
        // Exact integral = 9.0
        let y = [0.0, 1.0, 4.0, 9.0];
        let area = simpson_1d(&y);
        assert!((area - 9.0).abs() < 1e-10, "got {area}");
    }

    #[test]
    fn test_simpson_edge_cases() {
        assert!((simpson_1d(&[]) - 0.0).abs() < 1e-15);
        assert!((simpson_1d(&[5.0]) - 0.0).abs() < 1e-15);
        assert!((simpson_1d(&[2.0, 4.0]) - 3.0).abs() < 1e-15);
    }

    // ── loudness ─────────────────────────────────────────────────────

    #[test]
    fn test_k_weighting_coefficients() {
        let (b_shelf, a_shelf, b_hpass, a_hpass) = k_weighting_coefficients(8000.0);
        // Compare against known pyloudnorm values for 8kHz.
        assert!((b_shelf[0] - 1.32773315).abs() < 1e-5);
        assert!((a_shelf[0] - 1.0).abs() < 1e-10);
        assert!((b_hpass[0] - 0.97080775).abs() < 1e-5);
        assert!((a_hpass[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_lfilter_biquad_passthrough() {
        // Identity filter: b=[1,0,0], a=[1,0,0] should pass data unchanged.
        let b = [1.0, 0.0, 0.0];
        let a = [1.0, 0.0, 0.0];
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let out = lfilter_biquad(&b, &a, &data);
        for (x, y) in data.iter().zip(out.iter()) {
            assert!((x - y).abs() < 1e-10);
        }
    }

    #[test]
    fn test_k_weighted_squared_prefix_matches_two_pass_filter() {
        let data = [0.25_f32, -0.5, 0.75, -0.25, 0.1, -0.2];
        let input: Vec<f64> = data.iter().map(|&v| v as f64).collect();
        let (b_shelf, a_shelf, b_hpass, a_hpass) = k_weighting_coefficients(8000.0);

        let after_shelf = lfilter_biquad(&b_shelf, &a_shelf, &input);
        let filtered = lfilter_biquad(&b_hpass, &a_hpass, &after_shelf);
        let prefix = k_weighted_squared_prefix(&data, &b_shelf, &a_shelf, &b_hpass, &a_hpass);

        assert_eq!(prefix.len(), data.len() + 1);
        assert!(prefix[0].abs() < 1e-15);

        let mut expected = 0.0_f64;
        for (idx, value) in filtered.iter().enumerate() {
            expected += value * value;
            assert!(
                (prefix[idx + 1] - expected).abs() < 1e-10,
                "prefix mismatch at {idx}: {} != {expected}",
                prefix[idx + 1]
            );
        }
    }

    #[test]
    fn test_integrated_loudness_silence() {
        let silence = vec![0.0_f32; 8000];
        let lufs = integrated_loudness(&silence, 8000, 0.4);
        assert!(
            lufs.is_infinite() && lufs < 0.0,
            "silence should be -inf LUFS"
        );
    }

    #[test]
    fn test_integrated_loudness_sine() {
        // 1 second of 1kHz sine at 8kHz sample rate.
        let sr = 8000;
        let data: Vec<f32> = (0..sr)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / sr as f32).sin())
            .collect();
        let lufs = integrated_loudness(&data, sr as u32, 0.4);
        // A full-scale sine should be around -3 dBFS → roughly -3 LUFS.
        // The K-weighting will shift it somewhat. Just check it's in a sane range.
        assert!(
            lufs > -10.0 && lufs < 0.0,
            "sine LUFS={lufs} out of expected range"
        );
    }

    #[test]
    fn test_loudness_normalize_clips() {
        let data = [0.5_f32, -0.5, 0.8, -0.8];
        // Apply huge gain (+40 dB) to force clipping.
        let out = loudness_normalize(&data, -60.0, -20.0);
        for &v in &out {
            assert!(v >= -1.0 && v <= 1.0, "value {v} exceeds [-1, 1]");
        }
    }

    #[test]
    fn test_loudness_normalize_gain() {
        let data = [0.1_f32, -0.1];
        // +6 dB gain ≈ 2x.
        let out = loudness_normalize(&data, -22.0, -16.0);
        let expected_gain = 10.0_f64.powf(6.0 / 20.0); // ~1.995
        assert!((out[0] as f64 - 0.1 * expected_gain).abs() < 1e-4);
    }

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
