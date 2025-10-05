//! Real-time audio processing with CPAL

use crate::error::{CharonError, Result};
use crate::models::Model;
use crate::processor::Processor;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Stream, StreamConfig};
use ndarray::Array2;
use std::sync::{Arc, Mutex};

/// Real-time audio separator
pub struct RealtimeSeparator {
    model: Arc<Mutex<Model>>,
    processor: Arc<Mutex<Processor>>,
    buffer_size: usize,
    input_buffer: Arc<Mutex<Vec<f32>>>,
    output_buffers: Arc<Mutex<Vec<Vec<f32>>>>,
}

impl RealtimeSeparator {
    /// Create new real-time separator
    pub fn new(model: Model, processor: Processor, buffer_size: usize) -> Self {
        let num_sources = model.config().sources.len();
        Self {
            model: Arc::new(Mutex::new(model)),
            processor: Arc::new(Mutex::new(processor)),
            buffer_size,
            input_buffer: Arc::new(Mutex::new(Vec::with_capacity(buffer_size * 2))),
            output_buffers: Arc::new(Mutex::new(vec![Vec::new(); num_sources])),
        }
    }

    /// Start real-time processing stream
    pub fn start(&self) -> Result<Stream> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| CharonError::Audio("No input device available".to_string()))?;

        let config = device
            .default_input_config()
            .map_err(|e| CharonError::Audio(format!("Failed to get input config: {e}")))?;
        let stream = self.build_input_stream(&device, &config.into())?;

        stream
            .play()
            .map_err(|e| CharonError::Audio(format!("Failed to start stream: {e}")))?;
        Ok(stream)
    }

    fn build_input_stream(&self, device: &Device, config: &StreamConfig) -> Result<Stream> {
        let input_buffer = Arc::clone(&self.input_buffer);
        let output_buffers = Arc::clone(&self.output_buffers);
        let model = Arc::clone(&self.model);
        let _processor = Arc::clone(&self.processor);
        let buffer_size = self.buffer_size;
        let channels = config.channels as usize;

        let stream = device
            .build_input_stream(
                config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let mut buffer = input_buffer.lock().unwrap();
                    buffer.extend_from_slice(data);

                    if buffer.len() >= buffer_size * channels {
                        let process_data: Vec<f32> =
                            buffer.drain(..buffer_size * channels).collect();

                        let mut audio_array = Array2::zeros((channels, buffer_size));
                        for (i, &sample) in process_data.iter().enumerate() {
                            let ch = i % channels;
                            let samp = i / channels;
                            if samp < buffer_size {
                                audio_array[[ch, samp]] = sample;
                            }
                        }

                        if let Ok(model_lock) = model.lock() {
                            if let Ok(separated) = model_lock.infer(&audio_array) {
                                let mut out_buffers = output_buffers.lock().unwrap();
                                for (i, source) in separated.iter().enumerate() {
                                    if i < out_buffers.len() {
                                        out_buffers[i] = source.iter().copied().collect();
                                    }
                                }
                            }
                        }
                    }
                },
                |err| eprintln!("Stream error: {err}"),
                None,
            )
            .map_err(|e| CharonError::Audio(format!("Failed to build stream: {e}")))?;

        Ok(stream)
    }

    /// Get separated output for a specific source
    pub fn get_output(&self, source_index: usize) -> Option<Vec<f32>> {
        let buffers = self.output_buffers.lock().unwrap();
        buffers.get(source_index).cloned()
    }
}
