//! WebAssembly support for browser-based audio separation

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::{AudioBuffer, ModelConfig, Separator, SeparatorConfig};
use ndarray::Array2;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmSeparator {
    separator: Separator,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmSeparator {
    #[wasm_bindgen(constructor)]
    pub fn new(model_path: String, sample_rate: u32) -> Result<WasmSeparator, JsValue> {
        let config = SeparatorConfig {
            model: ModelConfig {
                model_path: model_path.into(),
                #[cfg(any(feature = "ort-backend", feature = "candle-backend"))]
                backend: None,
                sample_rate,
                channels: 2,
                sources: vec!["drums".to_string(), "bass".to_string(), "vocals".to_string(), "other".to_string()],
                chunk_size: Some(441000),
            },
            process: Default::default(),
        };

        let separator = Separator::new(config)
            .map_err(|e| JsValue::from_str(&format!("Failed to create separator: {}", e)))?;

        Ok(WasmSeparator { separator })
    }

    pub fn separate(&self, audio_data: Vec<f32>, channels: usize) -> Result<JsValue, JsValue> {
        let samples = audio_data.len() / channels;
        let mut array = Array2::zeros((channels, samples));
        
        for (i, &sample) in audio_data.iter().enumerate() {
            let ch = i % channels;
            let samp = i / channels;
            if samp < samples {
                array[[ch, samp]] = sample;
            }
        }

        let audio_buffer = AudioBuffer::new(array, self.separator.config.model.sample_rate);
        
        let stems = self.separator.separate(&audio_buffer)
            .map_err(|e| JsValue::from_str(&format!("Separation failed: {}", e)))?;

        serde_wasm_bindgen::to_value(&stems.sources)
            .map_err(|e| JsValue::from_str(&format!("Serialization failed: {}", e)))
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
}
