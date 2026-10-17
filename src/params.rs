//! JSON-friendly parameter objects for the high-level API.
//!
//! Fields are optional so absent keys retain the corresponding builder default.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::api::{
    Config, ConfigBuilder, ConfigBuilderError, ModelConfig, ModelConfigBuilder, SampleParams,
    SampleParamsBuilder, VideoConfig, VideoConfigBuilder,
};

/// JSON-friendly sampling parameters mirroring [SampleParams].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SampleParamsJson {
    pub sampling_method: Option<String>,
    pub steps: Option<i32>,
    pub eta: Option<f32>,
    pub scheduler: Option<String>,
    pub sigmas: Option<Vec<f32>>,
    pub cfg_scale: Option<f32>,
    pub img_cfg_scale: Option<f32>,
    pub guidance: Option<f32>,
    pub slg_scale: Option<f32>,
    pub skip_layer: Option<Vec<i32>>,
    pub skip_layer_start: Option<f32>,
    pub skip_layer_end: Option<f32>,
}

impl TryFrom<SampleParamsJson> for SampleParams {
    type Error = ConfigBuilderError;

    fn try_from(p: SampleParamsJson) -> Result<Self, Self::Error> {
        let mut b = SampleParamsBuilder::default();
        if let Some(v) = p.sampling_method {
            b.sampling_method(
                crate::api::sampler_from_name(&v)
                    .ok_or_else(|| ConfigBuilderError::ValidationError(format!("unknown sampling_method '{v}'")))?,
            );
        }
        if let Some(v) = p.steps {
            b.steps(v);
        }
        if let Some(v) = p.eta {
            b.eta(v);
        }
        if let Some(v) = p.scheduler {
            b.scheduler(
                crate::api::scheduler_from_name(&v)
                    .ok_or_else(|| ConfigBuilderError::ValidationError(format!("unknown scheduler '{v}'")))?,
            );
        }
        if let Some(v) = p.sigmas {
            b.sigmas(v);
        }
        if let Some(v) = p.cfg_scale {
            b.cfg_scale(v);
        }
        if let Some(v) = p.img_cfg_scale {
            b.img_cfg_scale(v);
        }
        if let Some(v) = p.guidance {
            b.guidance(v);
        }
        if let Some(v) = p.slg_scale {
            b.slg_scale(v);
        }
        if let Some(v) = p.skip_layer {
            b.skip_layer(v);
        }
        if let Some(v) = p.skip_layer_start {
            b.skip_layer_start(v);
        }
        if let Some(v) = p.skip_layer_end {
            b.skip_layer_end(v);
        }
        b.build()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelParams {
    pub model: Option<PathBuf>,
    pub diffusion_model: Option<PathBuf>,
    pub high_noise_diffusion_model: Option<PathBuf>,
    pub uncond_diffusion_model: Option<PathBuf>,
    pub clip_l: Option<PathBuf>,
    pub clip_g: Option<PathBuf>,
    pub clip_vision: Option<PathBuf>,
    pub t5xxl: Option<PathBuf>,
    pub llm: Option<PathBuf>,
    pub llm_vision: Option<PathBuf>,
    pub vae: Option<PathBuf>,
    pub audio_vae: Option<PathBuf>,
    pub taesd: Option<PathBuf>,
    pub control_net: Option<PathBuf>,
    pub photo_maker: Option<PathBuf>,
    pub pulid_weights: Option<PathBuf>,
    pub n_threads: Option<i32>,
    pub enable_mmap: Option<bool>,
    pub max_vram: Option<f32>,
    pub max_vram_spec: Option<String>,
    pub stream_layers: Option<bool>,
    pub eager_load: Option<bool>,
    pub flash_attention: Option<bool>,
    pub diffusion_flash_attention: Option<bool>,
    pub diffusion_conv_direct: Option<bool>,
    pub vae_conv_direct: Option<bool>,
    pub force_sdxl_vae_conv_scale: Option<bool>,
    pub backend: Option<String>,
    pub params_backend: Option<String>,
    pub split_mode: Option<String>,
    pub auto_fit: Option<bool>,
    pub rpc_servers: Option<String>,
    pub tensor_type_rules: Option<String>,
    pub model_args: Option<String>,
    /// Path to a Hugging Face `tokenizer.json`, or slot assignments such as
    /// `main=FILE,clip-l=FILE,clip-g=FILE`.
    ///
    /// A plain path is equivalent to `main=FILE`. Commas separate entries, so
    /// quote the value when a path contains spaces. Unknown keys, empty paths,
    /// malformed assignments, and duplicate slots are rejected.
    ///
    /// PiD and Lens require a `main` tokenizer; omitting this field keeps the
    /// embedded tokenizer for other models. T5/SentencePiece Unigram
    /// tokenizers are not supported.
    pub tokenizer: Option<String>,
    pub flow_shift: Option<f32>,
    pub timestep_shift: Option<i32>,
    pub circular: Option<bool>,
    pub circular_x: Option<bool>,
    pub circular_y: Option<bool>,
}

impl TryFrom<ModelParams> for ModelConfig {
    type Error = ConfigBuilderError;

    fn try_from(p: ModelParams) -> Result<Self, Self::Error> {
        let mut b = ModelConfigBuilder::default();
        macro_rules! set {
            ($field:ident) => {
                if let Some(v) = p.$field {
                    b.$field(v);
                }
            };
        }
        set!(model);
        set!(diffusion_model);
        set!(high_noise_diffusion_model);
        set!(uncond_diffusion_model);
        set!(clip_l);
        set!(clip_g);
        set!(clip_vision);
        set!(t5xxl);
        set!(llm);
        set!(llm_vision);
        set!(vae);
        set!(audio_vae);
        set!(taesd);
        set!(control_net);
        set!(photo_maker);
        set!(pulid_weights);
        set!(n_threads);
        set!(enable_mmap);
        set!(max_vram);
        set!(max_vram_spec);
        set!(stream_layers);
        set!(eager_load);
        set!(flash_attention);
        set!(diffusion_flash_attention);
        set!(diffusion_conv_direct);
        set!(vae_conv_direct);
        set!(force_sdxl_vae_conv_scale);
        set!(backend);
        set!(params_backend);
        set!(split_mode);
        set!(auto_fit);
        set!(rpc_servers);
        set!(tensor_type_rules);
        set!(model_args);
        set!(tokenizer);
        set!(flow_shift);
        set!(timestep_shift);
        set!(circular);
        set!(circular_x);
        set!(circular_y);
        b.build()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ImageParams {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub init_image: Option<PathBuf>,
    pub mask_image: Option<PathBuf>,
    pub control_image: Option<PathBuf>,
    pub ref_images: Vec<PathBuf>,
    pub output: Option<PathBuf>,
    pub preview_output: Option<PathBuf>,
    pub width: Option<i32>,
    pub height: Option<i32>,
    pub seed: Option<i64>,
    pub batch_count: Option<i32>,
    pub strength: Option<f32>,
    pub control_strength: Option<f32>,
    pub qwen_image_layers: Option<i32>,
    pub sample: Option<SampleParamsJson>,
    pub canny: Option<bool>,
    pub image_preprocess: Option<String>,
}

impl TryFrom<ImageParams> for Config {
    type Error = ConfigBuilderError;

    fn try_from(p: ImageParams) -> Result<Self, Self::Error> {
        let mut b = ConfigBuilder::default();
        b.prompt(p.prompt);
        if !p.ref_images.is_empty() {
            b.ref_images(p.ref_images);
        }
        macro_rules! set {
            ($field:ident) => {
                if let Some(v) = p.$field {
                    b.$field(v);
                }
            };
        }
        if let Some(v) = p.init_image {
            b.init_img(v);
        }
        if let Some(v) = p.mask_image {
            b.mask_img(v);
        }
        set!(control_image);
        set!(output);
        set!(preview_output);
        set!(negative_prompt);
        set!(width);
        set!(height);
        set!(seed);
        set!(batch_count);
        set!(strength);
        set!(control_strength);
        set!(qwen_image_layers);
        set!(canny);
        set!(image_preprocess);
        if let Some(sample) = p.sample {
            b.sample_params(SampleParams::try_from(sample)?);
        }
        b.build()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VideoParams {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub init_image: Option<PathBuf>,
    pub end_image: Option<PathBuf>,
    pub control_frames: Vec<PathBuf>,
    pub width: Option<i32>,
    pub height: Option<i32>,
    pub seed: Option<i64>,
    pub video_frames: Option<i32>,
    pub fps: Option<i32>,
    pub strength: Option<f32>,
    pub moe_boundary: Option<f32>,
    pub vace_strength: Option<f32>,
    pub sample: Option<SampleParamsJson>,
    pub high_noise_sample: Option<SampleParamsJson>,
    pub image_preprocess: Option<String>,
}

impl TryFrom<VideoParams> for VideoConfig {
    type Error = ConfigBuilderError;

    fn try_from(p: VideoParams) -> Result<Self, Self::Error> {
        let mut b = VideoConfigBuilder::default();
        b.prompt(p.prompt);
        if !p.control_frames.is_empty() {
            b.control_frames(p.control_frames);
        }
        if let Some(v) = p.init_image {
            b.init_img(v);
        }
        if let Some(v) = p.end_image {
            b.end_img(v);
        }
        macro_rules! set {
            ($field:ident) => {
                if let Some(v) = p.$field {
                    b.$field(v);
                }
            };
        }
        set!(negative_prompt);
        set!(width);
        set!(height);
        set!(seed);
        set!(video_frames);
        set!(fps);
        set!(strength);
        set!(moe_boundary);
        set!(vace_strength);
        set!(image_preprocess);
        if let Some(sample) = p.sample {
            b.sample_params(SampleParams::try_from(sample)?);
        }
        if let Some(high_noise_sample) = p.high_noise_sample {
            b.high_noise_sample_params(SampleParams::try_from(high_noise_sample)?);
        }
        b.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_params_use_builder_defaults_and_validation() {
        let image: ImageParams = serde_json::from_str(r#"{"prompt":"cat","width":768}"#).unwrap();
        assert!(Config::try_from(image).is_ok());

        let model: ModelParams = serde_json::from_str(
            r#"{"model":"model.gguf","eager_load":true,"model_args":"foo=bar","tokenizer":"main=main.json,clip-l=clip_l.json"}"#,
        )
        .unwrap();
        assert!(ModelConfig::try_from(model).is_ok());

        assert!(serde_json::from_str::<VideoParams>(r#"{"prompt":"x","unknown":1}"#).is_err());
    }
}
