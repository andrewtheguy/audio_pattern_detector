/// Format seconds as `HH:MM:SS.mmm`, rounding to the nearest millisecond.
pub fn seconds_to_time(seconds: f64) -> String {
    let milliseconds = seconds_to_ms(seconds);
    let minutes_remaining = milliseconds.div_euclid(60_000);
    let remaining_milliseconds = milliseconds.rem_euclid(60_000);
    let hours = minutes_remaining.div_euclid(60);
    let minutes = minutes_remaining.rem_euclid(60);
    format!(
        "{hours:02}:{minutes:02}:{:02}.{:03}",
        remaining_milliseconds / 1000,
        remaining_milliseconds % 1000
    )
}

/// Format seconds as `HH:MM:SS`, rounding to the nearest second.
pub fn seconds_to_time_whole(seconds: f64) -> String {
    let seconds = seconds.round_ties_even() as i64;
    let minutes_remaining = seconds.div_euclid(60);
    let remaining_seconds = seconds.rem_euclid(60);
    let hours = minutes_remaining.div_euclid(60);
    let minutes = minutes_remaining.rem_euclid(60);
    format!("{hours:02}:{minutes:02}:{remaining_seconds:02}")
}

/// Round seconds to integer milliseconds (ties to even).
pub fn seconds_to_ms(seconds: f64) -> i64 {
    (seconds * 1000.0).round_ties_even() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seconds_to_time() {
        assert_eq!(seconds_to_time(0.0), "00:00:00.000");
        assert_eq!(seconds_to_time(5.5), "00:00:05.500");
        assert_eq!(seconds_to_time(60.0), "00:01:00.000");
        assert_eq!(seconds_to_time(3661.0015), "01:01:01.002");
        assert_eq!(seconds_to_time(25.89875), "00:00:25.899");
        assert_eq!(seconds_to_time(360_000.0), "100:00:00.000");
    }

    #[test]
    fn test_seconds_to_time_whole() {
        assert_eq!(seconds_to_time_whole(0.0), "00:00:00");
        assert_eq!(seconds_to_time_whole(59.6), "00:01:00");
        assert_eq!(seconds_to_time_whole(2340.0), "00:39:00");
        assert_eq!(seconds_to_time_whole(3600.0), "01:00:00");
    }

    #[test]
    fn test_seconds_to_ms_rounds_ties_to_even() {
        assert_eq!(seconds_to_ms(0.0005), 0);
        assert_eq!(seconds_to_ms(0.0015), 2);
        assert_eq!(seconds_to_ms(1.407375), 1407);
    }
}
