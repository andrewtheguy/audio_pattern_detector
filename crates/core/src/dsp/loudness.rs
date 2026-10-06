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

/// Normalize audio in place to a target loudness in dB LUFS with hard
/// clipping.
///
/// Applies the gain needed to shift from `current_lufs` to `target_lufs`,
/// then hard-clips the output to [-1.0, 1.0].
pub fn loudness_normalize(data: &mut [f32], current_lufs: f64, target_lufs: f64) {
    let delta = target_lufs - current_lufs;
    let gain = 10.0_f64.powf(delta / 20.0);

    for x in data.iter_mut() {
        *x = ((*x as f64) * gain).clamp(-1.0, 1.0) as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let mut out = [0.5_f32, -0.5, 0.8, -0.8];
        // Apply huge gain (+40 dB) to force clipping.
        loudness_normalize(&mut out, -60.0, -20.0);
        for &v in &out {
            assert!((-1.0..=1.0).contains(&v), "value {v} exceeds [-1, 1]");
        }
    }

    #[test]
    fn test_loudness_normalize_in_place_exact() {
        // +20 dB is exactly a gain of 10; binary fractions keep the
        // products exact, and -0.125 * 10 is hard-clipped.
        let mut up = [0.0625_f32, -0.125, 0.03125, 0.0];
        loudness_normalize(&mut up, -36.0, -16.0);
        assert_eq!(up, [0.625, -1.0, 0.3125, 0.0]);

        // -20 dB attenuates by 10.
        let mut down = [0.5_f32, -1.0, 0.25];
        loudness_normalize(&mut down, 4.0, -16.0);
        for (actual, expected) in down.iter().zip([0.05_f32, -0.1, 0.025]) {
            assert!((actual - expected).abs() < 1e-7, "{actual} != {expected}");
        }

        // Zero gain change leaves every sample untouched, clipping aside.
        let mut same = [0.3_f32, -0.7, 1.5, -2.0];
        loudness_normalize(&mut same, -16.0, -16.0);
        assert_eq!(same, [0.3, -0.7, 1.0, -1.0]);

        // NaN input (silence after integrated_loudness) stays NaN.
        let mut silent = [0.0_f32, 0.0];
        loudness_normalize(&mut silent, f64::NEG_INFINITY, -16.0);
        assert!(silent.iter().all(|v| v.is_nan()), "{silent:?}");

        let mut empty: [f32; 0] = [];
        loudness_normalize(&mut empty, -20.0, -16.0);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_loudness_normalize_gain() {
        let mut out = [0.1_f32, -0.1];
        // +6 dB gain ≈ 2x.
        loudness_normalize(&mut out, -22.0, -16.0);
        let expected_gain = 10.0_f64.powf(6.0 / 20.0); // ~1.995
        assert!((out[0] as f64 - 0.1 * expected_gain).abs() < 1e-4);
    }
}
