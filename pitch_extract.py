import librosa
import numpy as np
import warnings

def extract_pitch(y, sr, hop_length, fmin, fmax):
    """
    Conduct pitch tracking. Return normalized pitch (0.0 ~ 1.0) and pitch mask where 1.0 corresponds to
    valid pitches and 0.0 corresponds to invalid pitches. 
    """
    y = y.numpy()
    pitches, _ = librosa.piptrack(y=y, sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax)
    pitches[pitches == 0.0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pitches = np.nanmin(pitches, axis=0)
    pitches[np.isnan(pitches)] = 0.0  # All-nan columns result in nan pitch. Use 0.0 for these values.
    pitch_mask = np.where(pitches > 0.0, 1.0, 0.0)
    normalized_pitches = np.clip((pitches - fmin) / (fmax - fmin), 0.0, 1.0)
    # normalized_pitches = torch.from_numpy(normalized_pitches)
    return normalized_pitches
