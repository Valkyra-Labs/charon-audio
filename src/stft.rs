//! Demucs spectrogram transforms (`HTDemucs._spec` / `_ispec`, demucs
//! 4.1.0), implemented with real FFTs.
//!
//! `_spec` reflect-pads the signal by `hop/2*3` on the left and enough on
//! the right to make the length a multiple of `hop`, runs `torch.stft`
//! (periodic Hann of `n_fft`, `normalized=True`, `center=True` with reflect
//! padding), drops the Nyquist bin and keeps frames `[2, 2 + le)` where
//! `le = ceil(length / hop)`. `_ispec` zero-pads the Nyquist bin and two
//! frames on each side, runs `torch.istft`, and trims `[pad, pad + length)`.
//! The transforms are verified against `torch.stft` fixtures in the tests.

use crate::error::{CharonError, Result};
use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;

/// A `[freqs, frames]` complex spectrogram of one channel, stored as
/// interleaved re/im per bin, frame-major: `data[frame * freqs + bin]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Spectrogram {
    pub freqs: usize,
    pub frames: usize,
    pub data: Vec<Complex<f32>>,
}

/// Demucs STFT/iSTFT for one configuration of `n_fft` and `hop`
pub struct DemucsStft {
    n_fft: usize,
    hop: usize,
    window: Vec<f32>,
    forward: Arc<dyn RealToComplex<f32>>,
    inverse: Arc<dyn ComplexToReal<f32>>,
}

impl DemucsStft {
    /// HTDemucs configuration: `n_fft` 4096, hop 1024
    pub fn htdemucs() -> Self {
        Self::new(4096, 1024)
    }

    pub fn new(n_fft: usize, hop: usize) -> Self {
        assert_eq!(hop, n_fft / 4, "Demucs requires hop == n_fft / 4");
        let mut planner = RealFftPlanner::<f32>::new();
        // torch.hann_window(n_fft) is periodic: 0.5 - 0.5 cos(2 pi n / N)
        let window = (0..n_fft)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n_fft as f64).cos())
            .map(|w| w as f32)
            .collect();
        Self {
            n_fft,
            hop,
            window,
            forward: planner.plan_fft_forward(n_fft),
            inverse: planner.plan_fft_inverse(n_fft),
        }
    }

    /// Number of frequency bins kept (`n_fft / 2`, Nyquist dropped)
    pub fn freqs(&self) -> usize {
        self.n_fft / 2
    }

    /// Number of frames produced for `length` samples: `ceil(length / hop)`
    pub fn frames(&self, length: usize) -> usize {
        length.div_ceil(self.hop)
    }

    /// `HTDemucs._spec` for one channel
    pub fn spec(&self, signal: &[f32]) -> Result<Spectrogram> {
        let length = signal.len();
        if length == 0 {
            return Err(CharonError::Processing("empty signal".to_string()));
        }
        let (n_fft, hop) = (self.n_fft, self.hop);
        let le = self.frames(length);
        let pad_left = hop / 2 * 3;
        let pad_right = pad_left + le * hop - length;
        // Demucs pad, then torch.stft's own centre pad of n_fft / 2, both reflect.
        let padded = reflect_pad(signal, pad_left + n_fft / 2, pad_right + n_fft / 2)?;

        let total_frames = (padded.len() - n_fft) / hop + 1;
        debug_assert_eq!(total_frames, le + 4);
        let freqs = self.freqs();
        let scale = 1.0 / (n_fft as f32).sqrt();
        let mut frame = self.forward.make_input_vec();
        let mut bins = self.forward.make_output_vec();
        let mut scratch = self.forward.make_scratch_vec();
        let mut data = Vec::with_capacity(freqs * le);
        for f in 2..2 + le {
            let start = f * hop;
            for (dst, (&x, &w)) in frame
                .iter_mut()
                .zip(padded[start..start + n_fft].iter().zip(&self.window))
            {
                *dst = x * w;
            }
            self.forward
                .process_with_scratch(&mut frame, &mut bins, &mut scratch)
                .map_err(|e| CharonError::Processing(e.to_string()))?;
            data.extend(bins[..freqs].iter().map(|c| c * scale));
        }
        Ok(Spectrogram {
            freqs,
            frames: le,
            data,
        })
    }

    /// `HTDemucs._ispec` for one channel: `spec` has `n_fft / 2` bins and
    /// `ceil(length / hop)` frames.
    pub fn ispec(&self, spec: &Spectrogram, length: usize) -> Result<Vec<f32>> {
        let (n_fft, hop) = (self.n_fft, self.hop);
        let le = self.frames(length);
        if spec.freqs != self.freqs() || spec.frames != le {
            return Err(CharonError::Processing(format!(
                "spectrogram is {}x{}, expected {}x{} for {length} samples",
                spec.freqs,
                spec.frames,
                self.freqs(),
                le
            )));
        }
        let pad = hop / 2 * 3;
        let out_len = hop * le + 2 * pad;
        // torch.istft with center=True works on out_len + n_fft samples and
        // trims n_fft / 2 from each end; frames [0, 2) and [le + 2, le + 4)
        // are the zero frames Demucs pads in.
        let total_frames = le + 4;
        let full_len = (total_frames - 1) * hop + n_fft;
        debug_assert_eq!(full_len, out_len + n_fft);
        let mut acc = vec![0.0f32; full_len];
        let mut env = vec![0.0f32; full_len];
        let scale = (n_fft as f32).sqrt() / n_fft as f32; // normalized=True, then the FFT's 1/N
        let mut bins = self.inverse.make_input_vec();
        let mut frame = self.inverse.make_output_vec();
        let mut scratch = self.inverse.make_scratch_vec();
        for f in 0..total_frames {
            let start = f * hop;
            for (i, w) in self.window.iter().enumerate() {
                env[start + i] += w * w;
            }
            if !(2..2 + le).contains(&f) {
                continue;
            }
            let src = &spec.data[(f - 2) * spec.freqs..(f - 1) * spec.freqs];
            bins[..spec.freqs].copy_from_slice(src);
            bins[0].im = 0.0;
            bins[spec.freqs] = Complex::new(0.0, 0.0); // Nyquist bin, dropped by _spec
            self.inverse
                .process_with_scratch(&mut bins, &mut frame, &mut scratch)
                .map_err(|e| CharonError::Processing(e.to_string()))?;
            for (i, (&x, &w)) in frame.iter().zip(&self.window).enumerate() {
                acc[start + i] += x * scale * w;
            }
        }
        let center = n_fft / 2;
        let out = (center..center + out_len)
            .map(|i| {
                // torch.istft asserts the envelope is above 1e-11 everywhere
                // it keeps; the Hann OLA at hop = n_fft / 4 satisfies that.
                acc[i] / env[i]
            })
            .skip(pad)
            .take(length)
            .collect();
        Ok(out)
    }
}

/// `torch.nn.functional.pad(mode="reflect")` for 1-D data
fn reflect_pad(x: &[f32], left: usize, right: usize) -> Result<Vec<f32>> {
    let n = x.len();
    if left >= n || right >= n {
        return Err(CharonError::Processing(format!(
            "signal of {n} samples is too short for reflect padding ({left}, {right})"
        )));
    }
    let mut out = Vec::with_capacity(left + n + right);
    out.extend((1..=left).rev().map(|i| x[i]));
    out.extend_from_slice(x);
    out.extend((1..=right).map(|i| x[n - 1 - i]));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture(name: &str) -> Vec<f32> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name);
        let bytes = std::fs::read(path).unwrap();
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }

    fn cac_to_spectrogram(cac: &[f32], freqs: usize, frames: usize) -> Spectrogram {
        // fixture layout: [2 (re, im), freqs, frames]
        let mut data = Vec::with_capacity(freqs * frames);
        for t in 0..frames {
            for f in 0..freqs {
                data.push(Complex::new(
                    cac[f * frames + t],
                    cac[freqs * frames + f * frames + t],
                ));
            }
        }
        Spectrogram {
            freqs,
            frames,
            data,
        }
    }

    #[test]
    fn reflect_pad_matches_torch() {
        let x = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(
            reflect_pad(&x, 2, 3).unwrap(),
            [3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0]
        );
        assert!(reflect_pad(&x, 4, 0).is_err());
    }

    #[test]
    fn spec_matches_torch_fixture() {
        let (n, freqs, frames) = (20000, 2048, 20);
        let input = fixture("stft_in.f32");
        assert_eq!(input.len(), n);
        let expected = cac_to_spectrogram(&fixture("stft_spec.f32"), freqs, frames);
        let stft = DemucsStft::htdemucs();
        let got = stft.spec(&input).unwrap();
        assert_eq!((got.freqs, got.frames), (freqs, frames));
        let max_err = got
            .data
            .iter()
            .zip(&expected.data)
            .map(|(a, b)| (a - b).norm())
            .fold(0.0f32, f32::max);
        let max_abs = expected
            .data
            .iter()
            .map(|c| c.norm())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 1e-5 * max_abs.max(1.0),
            "max error {max_err} (max |z| {max_abs})"
        );
    }

    #[test]
    fn ispec_matches_torch_fixture() {
        let (n, freqs, frames) = (20000, 2048, 20);
        let spec = cac_to_spectrogram(&fixture("istft_spec.f32"), freqs, frames);
        let expected = fixture("istft_out.f32");
        assert_eq!(expected.len(), n);
        let stft = DemucsStft::htdemucs();
        let got = stft.ispec(&spec, n).unwrap();
        assert_eq!(got.len(), n);
        let max_err = got
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_abs = expected.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(
            max_err < 1e-5 * max_abs.max(1.0),
            "max error {max_err} (max |x| {max_abs})"
        );
    }

    #[test]
    fn spec_ispec_round_trip_away_from_edges() {
        // Demucs' pair is not an identity at the edges (the two dropped
        // frames), but must be one in the interior.
        let n = 30000;
        let input: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.03).sin() * 0.5 + (i as f32 * 0.7).cos() * 0.2)
            .collect();
        let stft = DemucsStft::htdemucs();
        let back = stft.ispec(&stft.spec(&input).unwrap(), n).unwrap();
        let max_err = input[6000..n - 6000]
            .iter()
            .zip(&back[6000..n - 6000])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-4, "max interior error {max_err}");
    }
}
