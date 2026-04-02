#[cfg(feature = "python")]
mod python;

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

/// Keep only peaks whose prominence is at least `min_prominence`.
fn filter_by_prominence(data: &[f32], peaks: &mut Vec<usize>, min_prominence: f32) {
    peaks.retain(|&idx| compute_prominence(data, idx) >= min_prominence);
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
}
