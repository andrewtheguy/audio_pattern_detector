from scrape import get_sec

def test_get_sec():
    assert get_sec('00:59:59') == 3599
    assert get_sec('00:00:59') == 59
    assert get_sec('00:00:59.955') == 59.955
    assert get_sec('01:59:59.955') == 7199.955
    assert get_sec('00:00:01') == 1
    assert get_sec('00:00:01.006') == 1.006
    assert get_sec('00:00:01.0061') == 1.0061
    assert get_sec('00:00:01.0065') == 1.0065

