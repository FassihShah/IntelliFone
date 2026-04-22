import json
import numpy as np

SIDE_WEIGHTS = {
    "front": 1.0,
    "back": 0.8,
}

# Severity importance per damage type
CLASS_SEVERITY = {
    "crack": 8,
    "line": 7,
    "dot": 6,
}

# Scale factor for normalization (controls overall harshness)
SCALE = 10


def compute_condition_score(damage_data):
    # For loading file path inputs
    if isinstance(damage_data, str):
        with open(damage_data, 'r') as f:
            damage_data = json.load(f)

    damages = damage_data.get("damages", {})
    total_penalty = 0.0

    # ---- NEW: AI-detected flags ----
    ai_flags = {
        "screen_crack": False,
        "panel_dot": False,
        "panel_line": False
    }

    # Only front/back are used in the scoring flow now.
    for side, side_data in damages.items():
        if side not in SIDE_WEIGHTS or not side_data:
            continue

        side_weight = SIDE_WEIGHTS[side]

        for cls, detections in side_data.items():
            # Detect what damage types exist (boolean flags)
            if cls == "crack":
                ai_flags["screen_crack"] = True
            if cls == "dot":
                ai_flags["panel_dot"] = True
            if cls == "line":
                ai_flags["panel_line"] = True

            # Severity + magnitude scoring
            severity = CLASS_SEVERITY.get(cls, 5)
            magnitudes = [list(d.values())[0] for d in detections]
            total_magnitude = sum(magnitudes)

            # nonlinear penalty
            penalty = severity * side_weight * np.log1p(total_magnitude)
            total_penalty += penalty

    # Convert penalty to 0–20 score
    condition_score = max(20 - (total_penalty / SCALE), 0)

    return {
        "condition_score": float(round(condition_score, 2)),
        "penalty_total": float(round(total_penalty, 2)),

        # ---- NEW: AI detection flags ----
        "ai_detected": ai_flags
    }



# Example usage
if __name__ == "__main__":
    example = {
        "damages": {
            "front": {
                "dot": [{"area_px": 1158.8}],
                "crack": [{"length_px": 345.6}]
            }
        }
    }

    result = compute_condition_score(example)
    print(result)
