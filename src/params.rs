//! JSON-friendly parameter objects for the high-level API.
//!
//! Fields are optional so absent keys retain the corresponding builder default.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::api::{
    Config, ConfigBuilder, ConfigBuilderError, ModelConfig, ModelConfigBuilder, VideoConfig,
    VideoConfigBuilder,
};

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
    pub steps: Option<i32>,
    pub seed: Option<i64>,
    pub batch_count: Option<i32>,
    pub cfg_scale: Option<f32>,
    pub img_cfg_scale: Option<f32>,
    pub guidance: Option<f32>,
    pub strength: Option<f32>,
    pub eta: Option<f32>,
    pub control_strength: Option<f32>,
    pub qwen_image_layers: Option<i32>,
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
        set!(steps);
        set!(seed);
        set!(batch_count);
        set!(cfg_scale);
        set!(img_cfg_scale);
        set!(guidance);
        set!(strength);
        set!(eta);
        set!(control_strength);
        set!(qwen_image_layers);
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
    pub steps: Option<i32>,
    pub seed: Option<i64>,
    pub video_frames: Option<i32>,
    pub fps: Option<i32>,
    pub cfg_scale: Option<f32>,
    pub img_cfg_scale: Option<f32>,
    pub guidance: Option<f32>,
    pub strength: Option<f32>,
    pub eta: Option<f32>,
    pub high_noise_steps: Option<i32>,
    pub high_noise_cfg_scale: Option<f32>,
    pub high_noise_img_cfg_scale: Option<f32>,
    pub high_noise_guidance: Option<f32>,
    pub high_noise_eta: Option<f32>,
    pub high_noise_sigmas: Option<Vec<f32>>,
    pub high_noise_timestep_shift: Option<i32>,
    pub high_noise_flow_shift: Option<f32>,
    pub moe_boundary: Option<f32>,
    pub vace_strength: Option<f32>,
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
        set!(steps);
        set!(seed);
        set!(video_frames);
        set!(fps);
        set!(cfg_scale);
        set!(img_cfg_scale);
        set!(guidance);
        set!(strength);
        set!(eta);
        set!(high_noise_steps);
        set!(high_noise_cfg_scale);
        set!(high_noise_img_cfg_scale);
        set!(high_noise_guidance);
        set!(high_noise_eta);
        set!(high_noise_sigmas);
        set!(high_noise_timestep_shift);
        set!(high_noise_flow_shift);
        set!(moe_boundary);
        set!(vace_strength);
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
            r#"{"model":"model.gguf","eager_load":true,"model_args":"foo=bar"}"#,
        )
        .unwrap();
        assert!(ModelConfig::try_from(model).is_ok());

        assert!(serde_json::from_str::<VideoParams>(r#"{"prompt":"x","unknown":1}"#).is_err());
    }
}
