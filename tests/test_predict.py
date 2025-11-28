# Test cases for the predict module

import pytest
from app.predict import load_model, real_time_detect


def test_load_model():
    model, compiled = load_model('data/spam_fla_model.pkl')
    assert model is not None
    assert compiled is not None

def test_simple_spam_prediction():
    model, compiled = load_model('data/spam_fla_model.pkl')
    spammy = "Congratulations, you have won a free prize! Click here to claim your cash reward."
    res = real_time_detect(spammy, model, compiled)
    assert 'prediction' in res
    # At least one rule should match
    assert len(res['matched_rules']) >= 1