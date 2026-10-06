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
    // The height condition is applied while scanning: it depends only on the
    // peak's own value, so the result is the same as filtering afterwards,
    // and far fewer candidates are collected.
    let mut peaks = local_maxima_1d(data, options.height);

    if let Some(min_distance) = options.distance {
        filter_by_distance(data, &mut peaks, min_distance);
    }

    if let Some(min_prominence) = options.prominence {
        filter_by_prominence(data, &mut peaks, min_prominence);
    }

    peaks
}

/// Samples per block of the local-maxima scan; the candidate test for a
/// block is written as straight-line slice arithmetic so it vectorizes.
const SCAN_BLOCK: usize = 64;

/// Detect all local maxima in `data` whose value is at least `min_height`.
///
/// A sample is a local maximum when it is strictly greater than both its
/// immediate neighbours.  For plateaus (runs of identical values that are
/// higher than the values on both sides) the midpoint index (rounded down)
/// is returned, matching scipy's behaviour.
fn local_maxima_1d(data: &[f32], min_height: Option<f32>) -> Vec<usize> {
    let n = data.len();
    if n < 3 {
        return vec![];
    }
    let min_height = min_height.unwrap_or(f32::NEG_INFINITY);

    let mut peaks = Vec::new();
    let mut candidates = [false; SCAN_BLOCK];
    let mut start = 1;
    while start < n - 1 {
        let end = (start + SCAN_BLOCK).min(n - 1);
        let block = &mut candidates[..end - start];
        let previous = &data[start - 1..end - 1];
        let current = &data[start..end];
        let next = &data[start + 1..end + 1];

        // A candidate rises from the left, does not keep rising (strict peak
        // or plateau start) and is tall enough. A NaN neighbour fails the
        // comparison, just as it would fail the plateau check below.
        for (((flag, &p), &c), &nx) in block.iter_mut().zip(previous).zip(current).zip(next) {
            *flag = p < c && c >= nx && c >= min_height;
        }
        if block.iter().any(|&flag| flag) {
            for (offset, _) in block.iter().enumerate().filter(|(_, &flag)| flag) {
                let left_edge = start + offset;
                // Advance through equal values (plateau).
                let mut right_edge = left_edge;
                while right_edge + 1 < n && data[right_edge] == data[right_edge + 1] {
                    right_edge += 1;
                }
                // Confirm the right side drops.
                if right_edge + 1 < n && data[right_edge] > data[right_edge + 1] {
                    peaks.push((left_edge + right_edge) / 2);
                }
            }
        }
        start = end;
    }
    peaks
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
    for &val in &data[(peak_idx + 1)..] {
        if val < right_min {
            right_min = val;
        }
        if val > peak_val {
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── local_maxima_1d ──────────────────────────────────────────────

    #[test]
    fn test_local_maxima_simple() {
        let data = [0.0, 1.0, 0.0, 2.0, 0.0];
        assert_eq!(local_maxima_1d(&data, None), vec![1, 3]);
    }

    #[test]
    fn test_local_maxima_empty_and_short() {
        assert_eq!(local_maxima_1d(&[], None), Vec::<usize>::new());
        assert_eq!(local_maxima_1d(&[1.0], None), Vec::<usize>::new());
        assert_eq!(local_maxima_1d(&[1.0, 2.0], None), Vec::<usize>::new());
    }

    #[test]
    fn test_local_maxima_plateau_even() {
        // Plateau [1,1] spans indices 1..2 → midpoint = 1
        let data = [0.0, 1.0, 1.0, 0.0];
        assert_eq!(local_maxima_1d(&data, None), vec![1]);
    }

    #[test]
    fn test_local_maxima_plateau_odd() {
        // Plateau [1,1,1] spans indices 1..3 → midpoint = 2
        let data = [0.0, 1.0, 1.0, 1.0, 0.0];
        assert_eq!(local_maxima_1d(&data, None), vec![2]);
    }

    #[test]
    fn test_local_maxima_monotonic() {
        let ascending = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(local_maxima_1d(&ascending, None), Vec::<usize>::new());

        let descending = [5.0, 4.0, 3.0, 2.0, 1.0];
        assert_eq!(local_maxima_1d(&descending, None), Vec::<usize>::new());
    }

    #[test]
    fn test_local_maxima_height_in_scan() {
        let data = [0.0, 1.0, 0.0, 2.0, 2.0, 0.0, 3.0, 0.0];
        assert_eq!(local_maxima_1d(&data, None), vec![1, 3, 6]);
        assert_eq!(local_maxima_1d(&data, Some(2.0)), vec![3, 6]);
        assert_eq!(local_maxima_1d(&data, Some(2.5)), vec![6]);
        assert_eq!(local_maxima_1d(&data, Some(4.0)), Vec::<usize>::new());
    }

    #[test]
    fn test_local_maxima_across_scan_blocks() {
        // Peaks at the last and first positions of a block, a plateau that
        // starts in one block and ends in the next, and a NaN neighbour.
        let mut data = vec![0.0_f32; 3 * SCAN_BLOCK];
        data[SCAN_BLOCK] = 1.0; // start of block 2 (scan starts at index 1)
        data[SCAN_BLOCK + 1] = 1.0;
        data[2 * SCAN_BLOCK] = 2.0; // last position of block 2
        data[2 * SCAN_BLOCK + 1] = 2.0; // plateau continues into block 3
        data[2 * SCAN_BLOCK + 2] = 2.0;
        data[2 * SCAN_BLOCK + 10] = 5.0;
        data[2 * SCAN_BLOCK + 11] = f32::NAN;
        data[2 * SCAN_BLOCK + 20] = 4.0;
        assert_eq!(
            local_maxima_1d(&data, None),
            vec![SCAN_BLOCK, 2 * SCAN_BLOCK + 1, 2 * SCAN_BLOCK + 20]
        );
        assert_eq!(local_maxima_1d(&data, Some(1.5)), vec![2 * SCAN_BLOCK + 1, 2 * SCAN_BLOCK + 20]);
    }

    /// Plain one-pass scan with the height applied afterwards: the
    /// scipy `_local_maxima_1d` algorithm the block-wise scan must match.
    fn reference_local_maxima(data: &[f32], min_height: Option<f32>) -> Vec<usize> {
        let mut peaks = Vec::new();
        let mut i = 1;
        while i + 1 < data.len() {
            if data[i - 1] < data[i] {
                let left_edge = i;
                while i + 1 < data.len() && data[i] == data[i + 1] {
                    i += 1;
                }
                if i + 1 < data.len() && data[i] > data[i + 1] {
                    peaks.push((left_edge + i) / 2);
                }
            }
            i += 1;
        }
        if let Some(min_height) = min_height {
            peaks.retain(|&idx| data[idx] >= min_height);
        }
        peaks
    }

    /// Pseudo-random values with many repeats (so plateaus are common).
    fn repetitive_data(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i * 7919 % 23) as f32 - 11.0) / 4.0).collect()
    }

    #[test]
    fn test_local_maxima_matches_reference_scan() {
        let data = repetitive_data(1000);
        let expected = reference_local_maxima(&data, None);
        assert!(!expected.is_empty());
        assert_eq!(local_maxima_1d(&data, None), expected);
        let tall = reference_local_maxima(&data, Some(2.0));
        assert_ne!(tall.len(), expected.len());
        assert_eq!(local_maxima_1d(&data, Some(2.0)), tall);
    }

    #[test]
    fn test_local_maxima_matches_reference_for_every_length() {
        // Every length up to a few blocks, so the last block takes every
        // possible partial size and data ends at every block offset.
        for n in 0..=3 * SCAN_BLOCK + 3 {
            let data = repetitive_data(n);
            for min_height in [None, Some(-1.0), Some(0.0), Some(2.75), Some(3.0)] {
                assert_eq!(
                    local_maxima_1d(&data, min_height),
                    reference_local_maxima(&data, min_height),
                    "length {n}, min_height {min_height:?}"
                );
            }
        }
    }

    #[test]
    fn test_local_maxima_alternating_hits_every_block_position() {
        // 0,1,0,1,...: a peak at every odd index, so each position of
        // each block (first, last and interior) is a peak in one of the
        // two phases.
        let n = 3 * SCAN_BLOCK + 2;
        let odd_peaks: Vec<f32> = (0..n).map(|i| (i % 2) as f32).collect();
        let expected: Vec<usize> = (1..n - 1).filter(|i| i % 2 == 1).collect();
        assert_eq!(local_maxima_1d(&odd_peaks, None), expected);

        let even_peaks: Vec<f32> = (0..n).map(|i| ((i + 1) % 2) as f32).collect();
        let expected: Vec<usize> = (1..n - 1).filter(|i| i % 2 == 0).collect();
        assert_eq!(local_maxima_1d(&even_peaks, None), expected);
        assert_eq!(local_maxima_1d(&even_peaks, Some(1.5)), Vec::<usize>::new());
    }

    #[test]
    fn test_local_maxima_plateau_spanning_whole_block() {
        // A plateau longer than a block that starts in block 1 and drops in
        // block 3 is reported once, at its midpoint.
        let mut data = vec![0.0_f32; 4 * SCAN_BLOCK];
        let left = SCAN_BLOCK / 2;
        let right = left + SCAN_BLOCK + 10;
        data[left..=right].fill(1.0);
        assert_eq!(local_maxima_1d(&data, None), vec![(left + right) / 2]);
        assert_eq!(local_maxima_1d(&data, Some(1.0)), vec![(left + right) / 2]);
        assert_eq!(local_maxima_1d(&data, Some(1.1)), Vec::<usize>::new());

        // A plateau that runs to the end of the data never drops: no peak.
        data[right + 1..].fill(1.0);
        assert_eq!(local_maxima_1d(&data, None), Vec::<usize>::new());
    }

    #[test]
    fn test_local_maxima_nan_handling_matches_reference() {
        // NaN as the peak value, as the left neighbour and as the right
        // neighbour, plus NaN inside a plateau. None of them is a peak and
        // none hides a real peak elsewhere.
        let mut data = vec![0.0_f32; 40];
        data[2] = f32::NAN; // NaN "peak"
        data[6] = f32::NAN; // NaN left of a rise
        data[7] = 1.0;
        data[12] = 1.0;
        data[13] = f32::NAN; // NaN right of a peak
        data[20] = 1.0; // plateau broken by NaN
        data[21] = f32::NAN;
        data[22] = 1.0;
        data[30] = 2.0; // ordinary peak
        let expected = reference_local_maxima(&data, None);
        assert_eq!(expected, vec![30]);
        assert_eq!(local_maxima_1d(&data, None), expected);
        assert_eq!(local_maxima_1d(&data, Some(1.0)), vec![30]);
        assert_eq!(local_maxima_1d(&data, Some(f32::NAN)), Vec::<usize>::new());
    }

    #[test]
    fn test_find_peaks_height_matches_filtering_afterwards() {
        // find_peaks with height must equal find_peaks without height
        // followed by a height filter, including with distance and
        // prominence applied after it.
        let data = repetitive_data(3 * SCAN_BLOCK + 7);
        for height in [None, Some(0.5), Some(2.0)] {
            for distance in [None, Some(3)] {
                for prominence in [None, Some(1.0)] {
                    let with_height = find_peaks_1d(&data, &FindPeaksOptions { height, distance, prominence });
                    let mut unfiltered =
                        find_peaks_1d(&data, &FindPeaksOptions { height: None, distance: None, prominence: None });
                    if let Some(min_height) = height {
                        unfiltered.retain(|&idx| data[idx] >= min_height);
                    }
                    if let Some(min_distance) = distance {
                        filter_by_distance(&data, &mut unfiltered, min_distance);
                    }
                    if let Some(min_prominence) = prominence {
                        filter_by_prominence(&data, &mut unfiltered, min_prominence);
                    }
                    assert_eq!(
                        with_height, unfiltered,
                        "height {height:?} distance {distance:?} prominence {prominence:?}"
                    );
                }
            }
        }
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
}
