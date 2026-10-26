"""Online serving code (Module 9). Runs inside AWS Lambda, so it may import ONLY numpy and the standard library
(boto3 is imported lazily, and only where a real DynamoDB / SSM client is needed). No TensorFlow, xgboost, pandas, scipy or pymilvus here:
tests/test_serving_static.py enforces this.

LABELS that must travel with every response and demo:
  * the request clock is SIMULATED (MovieLens ends in 2019): clock = the user's last recorded activity + a gap chosen by the caller;
  * device is SIMULATED; the M3b model reacts to a PLANTED synthetic effect ("Synthetic device effect, pipeline demonstration, not a finding");
  * bandit feedback is SIMULATED ("SIMULATION, NOT AN A/B TEST")."""
LABEL_CLOCK = "Request clock is SIMULATED: the user's last recorded activity plus a chosen gap (MovieLens data ends in 2019)."
LABEL_DEVICE = "Device is SIMULATED: MovieLens has no device information."
LABEL_M3B = "Synthetic device effect, pipeline demonstration, not a finding"
LABEL_BANDIT = "SIMULATION, NOT AN A/B TEST"
