use std::collections::HashMap;
use std::ffi::CStr;
use std::ffi::CString;
use std::ffi::c_char;
use std::ffi::c_void;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::path::PathBuf;
use std::ptr::null;
use std::ptr::null_mut;
use std::slice;
use std::str::FromStr;
use std::sync::mpsc::Sender;
use std::sync::{Arc, OnceLock, RwLock};

use chrono::Local;
use derive_builder::Builder;
use diffusion_rs_sys::free_sd_audio;
use diffusion_rs_sys::free_upscaler_ctx;
use diffusion_rs_sys::generate_image;
use diffusion_rs_sys::generate_video;
use diffusion_rs_sys::new_upscaler_ctx;
use diffusion_rs_sys::sd_audio_t;
use diffusion_rs_sys::sd_cache_mode_t;
use diffusion_rs_sys::sd_cache_params_t;
use diffusion_rs_sys::sd_ctx_params_t;
use diffusion_rs_sys::sd_embedding_t;
use diffusion_rs_sys::sd_get_default_sample_method;
use diffusion_rs_sys::sd_get_default_scheduler;
use diffusion_rs_sys::sd_guidance_params_t;
use diffusion_rs_sys::sd_hires_params_t;
use diffusion_rs_sys::sd_image_t;
use diffusion_rs_sys::sd_img_gen_params_t;
use diffusion_rs_sys::sd_img_gen_params_to_str;
use diffusion_rs_sys::sd_log_level_t;
use diffusion_rs_sys::sd_lora_t;
use diffusion_rs_sys::sd_pm_params_t;
use diffusion_rs_sys::sd_sample_params_t;
use diffusion_rs_sys::sd_set_log_callback;
use diffusion_rs_sys::sd_set_preview_callback;
use diffusion_rs_sys::sd_set_progress_callback;
use diffusion_rs_sys::sd_slg_params_t;
use diffusion_rs_sys::sd_tiling_params_t;
use diffusion_rs_sys::sd_vid_gen_params_t;
use diffusion_rs_sys::upscaler_ctx_t;
use image::ImageBuffer;
use image::ImageError;
use image::RgbImage;
use libc::free;
use little_exif::exif_tag::ExifTag;
use little_exif::metadata::Metadata;
use thiserror::Error;
use walkdir::DirEntry;
use walkdir::WalkDir;

use diffusion_rs_sys::free_sd_ctx;
use diffusion_rs_sys::new_sd_ctx;
use diffusion_rs_sys::sd_ctx_t;

/// Specify the range function
pub use diffusion_rs_sys::rng_type_t as RngFunction;

/// Sampling methods
pub use diffusion_rs_sys::sample_method_t as SampleMethod;

/// Denoiser sigma schedule
pub use diffusion_rs_sys::scheduler_t as Scheduler;

/// Prediction override
pub use diffusion_rs_sys::prediction_t as Prediction;

/// Weight type
pub use diffusion_rs_sys::sd_type_t as WeightType;

/// Preview mode
pub use diffusion_rs_sys::preview_t as PreviewType;

/// Lora mode
pub use diffusion_rs_sys::lora_apply_mode_t as LoraModeType;

/// Hires mode
pub use diffusion_rs_sys::sd_hires_upscaler_t as Upscaler;

/// Log level reported by stable-diffusion.cpp
pub use diffusion_rs_sys::sd_log_level_t as LogLevel;

static VALID_EXT: [&str; 3] = ["gguf", "safetensors", "pt"];
type SdLogger = dyn Fn(LogLevel, &str) + Send + Sync + 'static;
static SD_LOGGER: OnceLock<RwLock<Option<Arc<SdLogger>>>> = OnceLock::new();

fn sd_logger() -> &'static RwLock<Option<Arc<SdLogger>>> {
    SD_LOGGER.get_or_init(|| RwLock::new(None))
}

unsafe extern "C" fn sd_log_callback(
    level: sd_log_level_t,
    text: *const c_char,
    _data: *mut c_void,
) {
    let logger = {
        let guard = sd_logger().read().unwrap();
        guard.as_ref().cloned()
    };

    let Some(logger) = logger else {
        return;
    };

    let message = unsafe {
        if text.is_null() {
            None
        } else {
            Some(CStr::from_ptr(text).to_string_lossy())
        }
    };

    let _ = catch_unwind(AssertUnwindSafe(|| {
        logger(level, message.as_deref().unwrap_or(""))
    }));
}

/// Install a process-global stable-diffusion.cpp logger callback.
///
/// The callback may be invoked from native worker threads and remains active
/// until replaced or cleared with [clear_sd_logger].
pub fn set_sd_logger<F>(logger: F)
where
    F: Fn(LogLevel, &str) + Send + Sync + 'static,
{
    {
        let mut guard = sd_logger().write().unwrap();
        *guard = Some(Arc::new(logger));
    }

    unsafe {
        sd_set_log_callback(Some(sd_log_callback), null_mut());
    }
}

/// Remove the process-global stable-diffusion.cpp logger callback.
pub fn clear_sd_logger() {
    {
        let mut guard = sd_logger().write().unwrap();
        *guard = None;
    }

    unsafe {
        sd_set_log_callback(None, null_mut());
    }
}

#[allow(unused)]
#[derive(Debug)]
/// Progress message returned fron [gen_img_with_progress]
pub struct Progress {
    pub step: i32,
    pub steps: i32,
    pub time: f32,
}

#[allow(unused)]
#[derive(Debug, Clone, PartialEq)]
/// Owned PCM audio buffer returned by video generation progress events.
pub struct AudioBuffer {
    pub sample_rate: u32,
    pub channels: u32,
    pub samples: Vec<f32>,
}

#[allow(unused)]
#[derive(Debug)]
pub enum ExtProgress {
    Progress(Progress),
    Preview(RgbImage),
    PreviewFrames(Vec<RgbImage>),
    FinalImage(RgbImage),
    FinalFrames(Vec<RgbImage>),
    FinalFramesWithAudio(Vec<RgbImage>, AudioBuffer),
}

#[non_exhaustive]
#[derive(Error, Debug)]
/// Error that can occurs while forwarding models
pub enum DiffusionError {
    #[error("The underling stablediffusion.cpp function returned NULL")]
    Forward,
    #[error(transparent)]
    StoreImages(#[from] ImageError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("The underling upscaler model returned a NULL image")]
    Upscaler,
    #[error(transparent)]
    Utf8(#[from] std::ffi::IntoStringError),
}

#[repr(i32)]
#[non_exhaustive]
#[derive(Debug, Default, Copy, Clone, Hash, PartialEq, Eq)]
/// Ignore the lower X layers of CLIP network
pub enum ClipSkip {
    /// Will be [ClipSkip::None] for SD1.x, [ClipSkip::OneLayer] for SD2.x
    #[default]
    Unspecified = 0,
    None = 1,
    OneLayer = 2,
}

#[derive(Debug, Clone)]
/// Image source used by generation configs
pub enum InputImage {
    Path(PathBuf),
    Image(RgbImage),
}

impl Default for InputImage {
    fn default() -> Self {
        Self::Path(PathBuf::new())
    }
}

impl From<PathBuf> for InputImage {
    fn from(value: PathBuf) -> Self {
        Self::Path(value)
    }
}

impl From<&Path> for InputImage {
    fn from(value: &Path) -> Self {
        Self::Path(value.to_path_buf())
    }
}

impl From<&str> for InputImage {
    fn from(value: &str) -> Self {
        Self::Path(PathBuf::from(value))
    }
}

impl From<String> for InputImage {
    fn from(value: String) -> Self {
        Self::Path(PathBuf::from(value))
    }
}

impl From<RgbImage> for InputImage {
    fn from(value: RgbImage) -> Self {
        Self::Image(value)
    }
}

impl InputImage {
    fn is_present(&self) -> bool {
        match self {
            Self::Path(path) => !path.as_os_str().is_empty(),
            Self::Image(_) => true,
        }
    }
}

type EmbeddingsStorage = (PathBuf, Vec<(CLibString, CLibPath)>, Vec<sd_embedding_t>);
type LoraStorage = Vec<(CLibPath, LoraSpec)>;

/// Specify the instructions for a Lora model
#[derive(Default, Debug, Clone)]
pub struct LoraSpec {
    pub file_name: String,
    pub is_high_noise: bool,
    pub multiplier: f32,
}

/// Parameters for Spectrum Caching
#[derive(Builder, Debug, Clone)]
pub struct SpectrumCacheParams {
    /// Chebyshev vs Taylor blend weight (0=Taylor, 1=Chebyshev)
    #[builder(default = "0.40")]
    w: f32,

    /// Chebyshev polynomial degree
    #[builder(default = "3")]
    m: i32,

    /// Ridge regression regularization
    #[builder(default = "1.0")]
    lam: f32,

    /// Initial window size (compute every N steps)
    #[builder(default = "2")]
    window: i32,

    /// Window growth per computed step after warmup
    #[builder(default = "0.50")]
    flex: f32,

    /// Steps to always compute before caching starts
    #[builder(default = "4")]
    warmup: i32,

    /// Stop caching at this fraction of total steps
    #[builder(default = "0.9")]
    stop: f32,
}

/// Parameters for UCache
#[derive(Builder, Debug, Clone)]
pub struct UCacheParams {
    /// Error threshold for reuse decision
    #[builder(default = "1.0")]
    threshold: f32,

    /// Start caching at this percent of steps
    #[builder(default = "0.15")]
    start: f32,

    /// Stop caching at this percent of steps
    #[builder(default = "0.95")]
    end: f32,

    /// Error decay rate (0-1)
    #[builder(default = "1.0")]
    decay: f32,

    /// Scale threshold by output norm
    #[builder(default = "true")]
    relative: bool,

    /// Reset error after computing
    /// true: Resets accumulated error after each computed step. More aggressive caching, works well with most samplers.
    /// false: Keeps error accumulated. More conservative, recommended for euler_a sampler
    #[builder(default = "true")]
    reset: bool,
}

/// Parameters for Easy Cache
#[derive(Builder, Debug, Clone)]
pub struct EasyCacheParams {
    /// Error threshold for reuse decision
    #[builder(default = "0.2")]
    threshold: f32,

    /// Start caching at this percent of steps
    #[builder(default = "0.15")]
    start: f32,

    /// Stop caching at this percent of steps
    #[builder(default = "0.95")]
    end: f32,
}

/// Parameters for Db Cache
#[derive(Builder, Debug, Clone)]
pub struct DbCacheParams {
    /// Front blocks to always compute
    #[builder(default = "8")]
    fn_blocks: i32,

    /// Back blocks to always compute
    #[builder(default = "0")]
    bn_blocks: i32,

    /// L1 residual difference threshold
    #[builder(default = "0.08")]
    threshold: f32,

    /// Steps before caching starts
    #[builder(default = "8")]
    warmup: i32,

    /// Steps Computation Mask controls which steps can be cached
    /// E.g.: "1,1,1,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,1"
    /// where 1 means compute, 0 means cache
    #[builder(default = "CLibString::default()")]
    scm_mask: CLibString,

    /// Scm Policy
    #[builder(default = "ScmPolicy::default()")]
    scm_policy_dynamic: ScmPolicy,
}

/// Steps Computation Mask Policy controls when to cache steps
#[derive(Debug, Default, Clone)]
pub enum ScmPolicy {
    /// Always cache on cacheable steps
    Static,
    #[default]
    /// Check threshold before caching
    Dynamic,
}

/// Hires parameters
#[derive(Builder, Debug, Clone)]
pub struct HiresParams {
    /// highres fix target width, 0 to use scale (default: 0)
    #[builder(default = "0")]
    width: i32,
    /// highres fix target height, 0 to use scale (default: 0)
    #[builder(default = "0")]
    height: i32,
    /// highres fix target width, 0 to use scale (default: 0)
    #[builder(default = "0")]
    steps: i32,
    /// highres fix upscaler tile size, reserved for model-backed upscalers (default: 128)
    #[builder(default = "128")]
    upscale_tile_size: i32,
    /// highres fix scale when sizes is not set (default: 2.0)
    #[builder(default = "2.0")]
    scale: f32,
    /// highres fix second pass denoising strength (default: 0.7)
    #[builder(default = "0.7")]
    denoising_strength: f32,

    /// Custom sigma values for the highres pass
    #[builder(default = "Default::default()")]
    custom_sigmas: Vec<f32>,
}

/// Config struct for a specific diffusion model
#[derive(Builder, Debug)]
#[builder(
    setter(into, strip_option),
    build_fn(error = "ConfigBuilderError", validate = "Self::validate")
)]
pub struct ModelConfig {
    /// Number of threads to use during computation (default: 0).
    /// If n_ threads <= 0, then threads will be set to the number of CPU physical cores.
    #[builder(default = "num_cpus::get_physical() as i32", setter(custom))]
    n_threads: i32,

    /// Whether to memory-map model (default: false)
    #[builder(default = "false")]
    enable_mmap: bool,

    /// Place the weights in RAM to save VRAM, and automatically load them into VRAM when needed
    #[builder(default = "false")]
    offload_params_to_cpu: bool,

    /// Maximum VRAM budget in GiB for graph-cut segmented execution. 0 disables graph splitting; -1 auto-detects free VRAM minus 1 GiB
    #[builder(default = "-1.0")]
    max_vram: f32,

    /// Path to esrgan model. Upscale images after generate, just RealESRGAN_x4plus_anime_6B supported by now
    #[builder(default = "Default::default()")]
    upscale_model: Option<CLibPath>,

    /// Run the ESRGAN upscaler this many times (default 1)
    #[builder(default = "1")]
    upscale_repeats: i32,

    /// Tile size for ESRGAN upscaler (default 128)
    #[builder(default = "128")]
    upscale_tile_size: i32,

    /// Path to full model
    #[builder(default = "Default::default()")]
    model: CLibPath,

    /// Path to the standalone diffusion model
    #[builder(default = "Default::default()")]
    diffusion_model: CLibPath,

    /// Path to the qwen2vl text encoder
    #[builder(default = "Default::default()")]
    llm: CLibPath,

    /// Path to the qwen2vl vit
    #[builder(default = "Default::default()")]
    llm_vision: CLibPath,

    /// Path to the clip-l text encoder
    #[builder(default = "Default::default()")]
    clip_l: CLibPath,

    /// Path to the clip-g text encoder
    #[builder(default = "Default::default()")]
    clip_g: CLibPath,

    /// Path to the clip-vision encoder
    #[builder(default = "Default::default()")]
    clip_vision: CLibPath,

    /// Path to the t5xxl text encoder
    #[builder(default = "Default::default()")]
    t5xxl: CLibPath,

    /// Path to vae
    #[builder(default = "Default::default()")]
    vae: CLibPath,

    /// Path to audio vae
    #[builder(default = "Default::default()")]
    audio_vae: CLibPath,

    /// Path to taesd. Using Tiny AutoEncoder for fast decoding (low quality)
    #[builder(default = "Default::default()")]
    taesd: CLibPath,

    /// Path to control net model
    #[builder(default = "Default::default()")]
    control_net: CLibPath,

    /// Path to embeddings
    #[builder(default = "Default::default()", setter(custom))]
    embeddings: EmbeddingsStorage,

    /// Path to embeddings connectors
    #[builder(default = "Default::default()")]
    embeddings_connectors: CLibPath,

    /// Path to PHOTOMAKER model
    #[builder(default = "Default::default()")]
    photo_maker: CLibPath,

    /// Path to PHOTOMAKER v2 id embed
    #[builder(default = "Default::default()")]
    pm_id_embed_path: CLibPath,

    /// Weight type. If not specified, the default is the type of the weight file
    #[builder(default = "WeightType::SD_TYPE_COUNT")]
    weight_type: WeightType,

    /// Lora model directory
    #[builder(default = "Default::default()", setter(custom))]
    lora_models: LoraStorage,

    /// Path to the standalone high noise diffusion model
    #[builder(default = "Default::default()")]
    high_noise_diffusion_model: CLibPath,

    /// Process vae in tiles to reduce memory usage (default: false)
    #[builder(default = "false")]
    vae_tiling: bool,

    /// Tile size for vae tiling (default: 32x32)
    #[builder(default = "(32,32)")]
    vae_tile_size: (i32, i32),

    /// Relative tile size for vae tiling, in fraction of image size if < 1, in number of tiles per dim if >=1 (overrides vae_tile_size)
    #[builder(default = "(0.,0.)")]
    vae_relative_tile_size: (f32, f32),

    /// Tile overlap for vae tiling, in fraction of tile size (default: 0.5)
    #[builder(default = "0.5")]
    vae_tile_overlap: f32,

    /// Enable temporal vae tiling
    #[builder(default = "false")]
    vae_temporal_tiling: bool,

    /// Extra vae tiling arguments, in backend-specific string format
    #[builder(default = "CLibString::default()")]
    vae_extra_tiling_args: CLibString,

    /// RNG (default: CUDA)
    #[builder(default = "RngFunction::CUDA_RNG")]
    rng: RngFunction,

    /// Sampler RNG. If [RngFunction::RNG_TYPE_COUNT] is used will default to rng value. (default: [RngFunction::RNG_TYPE_COUNT])",
    #[builder(default = "RngFunction::RNG_TYPE_COUNT")]
    sampler_rng_type: RngFunction,

    /// Denoiser sigma schedule (default: [Scheduler::SCHEDULER_COUNT]).
    /// Will default to [Scheduler::EXPONENTIAL_SCHEDULER] if a denoiser is already instantiated.
    /// Otherwise, [Scheduler::DISCRETE_SCHEDULER] is used.
    #[builder(default = "Scheduler::SCHEDULER_COUNT")]
    scheduler: Scheduler,

    /// Custom sigma values for the sampler
    #[builder(default = "Default::default()")]
    sigmas: Vec<f32>,

    /// Prediction type override (default: PREDICTION_COUNT)
    #[builder(default = "Prediction::PREDICTION_COUNT")]
    prediction: Prediction,

    /// Keep vae in cpu (for low vram) (default: false)
    #[builder(default = "false")]
    vae_on_cpu: bool,

    /// keep clip in cpu (for low vram) (default: false)
    #[builder(default = "false")]
    clip_on_cpu: bool,

    /// Keep controlnet in cpu (for low vram) (default: false)
    #[builder(default = "false")]
    control_net_cpu: bool,

    /// Use flash attention to reduce memory usage (model only).
    /// For most backends, it slows things down, but for cuda it generally speeds it up too. At the moment, it is only supported for some models and some backends (like cpu, cuda/rocm, metal).
    #[builder(default = "false")]
    diffusion_flash_attention: bool,

    /// Use flash attention to reduce memory usage.
    /// For most backends, it slows things down, but for cuda it generally speeds it up too. At the moment, it is only supported for some models and some backends (like cpu, cuda/rocm, metal).
    #[builder(default = "false")]
    flash_attention: bool,

    /// Disable dit mask for chroma
    #[builder(default = "false")]
    chroma_disable_dit_mask: bool,

    /// Enable t5 mask for chroma
    #[builder(default = "false")]
    chroma_enable_t5_mask: bool,

    /// t5 mask pad size of chroma
    #[builder(default = "1")]
    chroma_t5_mask_pad: i32,

    /// Use qwen image zero cond true optimization
    #[builder(default = "false")]
    use_qwen_image_zero_cond_true: bool,

    /// Use Conv2d direct in the diffusion model
    /// This might crash if it is not supported by the backend.
    #[builder(default = "false")]
    diffusion_conv_direct: bool,

    /// Use Conv2d direct in the vae model (should improve the performance)
    /// This might crash if it is not supported by the backend.
    #[builder(default = "false")]
    vae_conv_direct: bool,

    /// Force use of conv scale on sdxl vae
    #[builder(default = "false")]
    force_sdxl_vae_conv_scale: bool,

    /// Shift value for Flow models like SD3.x or WAN (default: auto)
    #[builder(default = "f32::INFINITY")]
    flow_shift: f32,

    /// Shift timestep for NitroFusion models, default: 0, recommended N for NitroSD-Realism around 250 and 500 for NitroSD-Vibrant
    #[builder(default = "0")]
    timestep_shift: i32,

    /// Prevents usage of taesd for decoding the final image
    #[builder(default = "false")]
    taesd_preview_only: bool,

    /// In auto mode, if the model weights contain any quantized parameters, the at_runtime mode will be used; otherwise, immediately will be used.The immediate mode may have precision and compatibility issues with quantized parameters, but it usually offers faster inference speed and, in some cases, lower memory usage. The at_runtime mode, on the other hand, is exactly the opposite
    #[builder(default = "LoraModeType::LORA_APPLY_AUTO")]
    lora_apply_mode: LoraModeType,

    /// Enable circular padding for convolutions
    #[builder(default = "false")]
    circular: bool,

    /// Enable circular RoPE wrapping on x-axis (width) only
    #[builder(default = "false")]
    circular_x: bool,

    /// Enable circular RoPE wrapping on y-axis (height) only
    #[builder(default = "false")]
    circular_y: bool,

    #[builder(default = "None", private)]
    upscaler_ctx: Option<*mut upscaler_ctx_t>,

    #[builder(default = "None", private)]
    diffusion_ctx: Option<(*mut sd_ctx_t, sd_ctx_params_t)>,

    /// Hires fix parameters and upscaler model.
    #[builder(default = "Self::hires_init()", setter(custom))]
    hires_params: (Upscaler, HiresParams, Option<CLibPath>),

    /// Extra parameters for sampling, currently used for SDXL sample params, in json string format
    #[builder(default = "CLibString::default()")]
    extra_sample_params: CLibString,

    /// Backend name for model computation
    #[builder(default = "CLibString::default()")]
    backend: CLibString,

    /// Backend name for parameter storage/loading
    #[builder(default = "CLibString::default()")]
    params_backend: CLibString,
}

impl ModelConfigBuilder {
    fn validate(&self) -> Result<(), ConfigBuilderError> {
        self.validate_model()
    }

    fn validate_model(&self) -> Result<(), ConfigBuilderError> {
        self.model
            .as_ref()
            .or(self.diffusion_model.as_ref())
            .map(|_| ())
            .ok_or(ConfigBuilderError::UninitializedField(
                "Model OR DiffusionModel must be valorized",
            ))
    }

    fn filter_valid_extensions(path: &Path) -> impl Iterator<Item = DirEntry> {
        WalkDir::new(path)
            .into_iter()
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext_str| VALID_EXT.contains(&ext_str))
                    .unwrap_or(false)
            })
    }
    fn build_single_lora_storage(
        spec: &LoraSpec,
        valid_loras: &HashMap<String, PathBuf>,
    ) -> (CLibPath, LoraSpec) {
        let path = valid_loras.get(&spec.file_name).unwrap().as_path();
        let c_path = CLibPath::from(path);
        (c_path, spec.clone())
    }

    pub fn embeddings(&mut self, embeddings_dir: &Path) -> &mut Self {
        let data: Vec<(CLibString, CLibPath)> = Self::filter_valid_extensions(embeddings_dir)
            .map(|entry| {
                let file_stem = entry
                    .path()
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .unwrap_or_default()
                    .to_owned();
                (CLibString::from(file_stem), CLibPath::from(entry.path()))
            })
            .collect();
        let data_pointer = data
            .iter()
            .map(|(name, path)| sd_embedding_t {
                name: name.as_ptr(),
                path: path.as_ptr(),
            })
            .collect();
        self.embeddings = Some((embeddings_dir.to_path_buf(), data, data_pointer));
        self
    }

    pub fn lora_models(&mut self, lora_model_dir: &Path, specs: Vec<LoraSpec>) -> &mut Self {
        let valid_loras: HashMap<String, PathBuf> = Self::filter_valid_extensions(lora_model_dir)
            .map(|entry| {
                let path = entry.path();
                (
                    path.file_stem()
                        .and_then(|stem| stem.to_str())
                        .unwrap_or_default()
                        .to_owned(),
                    path.to_path_buf(),
                )
            })
            .collect();
        let valid_lora_names: Vec<&String> = valid_loras.keys().collect();
        let standard = specs
            .iter()
            .filter(|s| valid_lora_names.contains(&&s.file_name) && !s.is_high_noise)
            .map(|s| Self::build_single_lora_storage(s, &valid_loras));
        let high_noise = specs
            .iter()
            .filter(|s| valid_lora_names.contains(&&s.file_name) && s.is_high_noise)
            .map(|s| Self::build_single_lora_storage(s, &valid_loras));

        self.lora_models_internal(standard.chain(high_noise).collect())
    }

    fn lora_models_internal(&mut self, lora_storage: LoraStorage) -> &mut Self {
        self.lora_models = Some(lora_storage);
        self
    }

    pub fn n_threads(&mut self, value: i32) -> &mut Self {
        self.n_threads = if value > 0 {
            Some(value)
        } else {
            Some(num_cpus::get_physical() as i32)
        };
        self
    }

    pub fn hires_params(
        &mut self,
        upscaler: Upscaler,
        params: HiresParams,
        custom_model: Option<&Path>,
    ) -> &mut Self {
        if upscaler == Upscaler::SD_HIRES_UPSCALER_COUNT
            || (upscaler == Upscaler::SD_HIRES_UPSCALER_MODEL && custom_model.is_none())
        {
            panic!("Invalid combination for {upscaler:?} and {custom_model:?}")
        }
        self.hires_params = Some((upscaler, params, custom_model.map(Into::into)));

        self
    }

    fn hires_init() -> (Upscaler, HiresParams, Option<CLibPath>) {
        (
            Upscaler::SD_HIRES_UPSCALER_NONE,
            HiresParamsBuilder::default().build().unwrap(),
            None,
        )
    }
}

impl ModelConfig {
    unsafe fn upscaler_ctx(&mut self) -> Option<*mut upscaler_ctx_t> {
        unsafe {
            if self.upscale_model.is_none() || self.upscale_repeats == 0 {
                None
            } else {
                if self.upscaler_ctx.is_none() {
                    let upscaler = new_upscaler_ctx(
                        self.upscale_model.as_ref().unwrap().as_ptr(),
                        self.offload_params_to_cpu,
                        self.diffusion_conv_direct,
                        self.n_threads,
                        self.upscale_tile_size,
                        self.backend.as_optional_ptr(),
                        self.params_backend.as_optional_ptr(),
                    );
                    self.upscaler_ctx = Some(upscaler);
                }
                self.upscaler_ctx
            }
        }
    }

    unsafe fn diffusion_ctx(&mut self, vae_decode_only: bool) -> *mut sd_ctx_t {
        unsafe {
            // This is required to support img2img after text2img generation
            // otherwise the context is cached and won't have a decode graph
            // leading to an assertion error in sdcpp
            if let Some((sd_ctx, sd_ctx_params)) = self.diffusion_ctx.as_ref()
                && sd_ctx_params.vae_decode_only != vae_decode_only
            {
                sd_set_progress_callback(None, null_mut());
                free_sd_ctx(*sd_ctx);
                self.diffusion_ctx = None;
            }
            if self.diffusion_ctx.is_none() {
                let sd_ctx_params = sd_ctx_params_t {
                    model_path: self.model.as_ptr(),
                    llm_path: self.llm.as_ptr(),
                    llm_vision_path: self.llm_vision.as_ptr(),
                    clip_l_path: self.clip_l.as_ptr(),
                    clip_g_path: self.clip_g.as_ptr(),
                    clip_vision_path: self.clip_vision.as_ptr(),
                    high_noise_diffusion_model_path: self.high_noise_diffusion_model.as_ptr(),
                    t5xxl_path: self.t5xxl.as_ptr(),
                    diffusion_model_path: self.diffusion_model.as_ptr(),
                    embeddings_connectors_path: self.embeddings_connectors.as_ptr(),
                    vae_path: self.vae.as_ptr(),
                    audio_vae_path: self.audio_vae.as_ptr(),
                    taesd_path: self.taesd.as_ptr(),
                    control_net_path: self.control_net.as_ptr(),
                    embeddings: self.embeddings.2.as_ptr(),
                    embedding_count: self.embeddings.1.len() as u32,
                    photo_maker_path: self.photo_maker.as_ptr(),
                    vae_decode_only,
                    free_params_immediately: false,
                    n_threads: self.n_threads,
                    wtype: self.weight_type,
                    rng_type: self.rng,
                    keep_clip_on_cpu: self.clip_on_cpu,
                    keep_control_net_on_cpu: self.control_net_cpu,
                    keep_vae_on_cpu: self.vae_on_cpu,
                    diffusion_flash_attn: self.diffusion_flash_attention,
                    flash_attn: self.flash_attention,
                    diffusion_conv_direct: self.diffusion_conv_direct,
                    chroma_use_dit_mask: !self.chroma_disable_dit_mask,
                    chroma_use_t5_mask: self.chroma_enable_t5_mask,
                    chroma_t5_mask_pad: self.chroma_t5_mask_pad,
                    vae_conv_direct: self.vae_conv_direct,
                    offload_params_to_cpu: self.offload_params_to_cpu,
                    prediction: self.prediction,
                    force_sdxl_vae_conv_scale: self.force_sdxl_vae_conv_scale,
                    tae_preview_only: self.taesd_preview_only,
                    lora_apply_mode: self.lora_apply_mode,
                    tensor_type_rules: null(),
                    sampler_rng_type: self.sampler_rng_type,
                    circular_x: self.circular || self.circular_x,
                    circular_y: self.circular || self.circular_y,
                    qwen_image_zero_cond_t: self.use_qwen_image_zero_cond_true,
                    enable_mmap: self.enable_mmap,
                    max_vram: self.max_vram,
                    backend: self.backend.as_optional_ptr(),
                    params_backend: self.params_backend.as_optional_ptr(),
                };
                let ctx = new_sd_ctx(&sd_ctx_params);
                self.diffusion_ctx = Some((ctx, sd_ctx_params))
            }
            self.diffusion_ctx.unwrap().0
        }
    }
}

impl Drop for ModelConfig {
    fn drop(&mut self) {
        //Cleanup CTX section
        unsafe {
            if let Some((sd_ctx, _)) = self.diffusion_ctx {
                free_sd_ctx(sd_ctx);
            }

            if let Some(upscaler_ctx) = self.upscaler_ctx {
                free_upscaler_ctx(upscaler_ctx);
            }
        }
    }
}

impl From<&ModelConfig> for ModelConfigBuilder {
    fn from(value: &ModelConfig) -> Self {
        let mut builder = ModelConfigBuilder::default();
        let hires_path = value
            .hires_params
            .2
            .clone()
            .map(|f| PathBuf::from_str(&f.0.into_string().unwrap()).unwrap());
        builder
            .n_threads(value.n_threads)
            .max_vram(value.max_vram)
            .offload_params_to_cpu(value.offload_params_to_cpu)
            .upscale_repeats(value.upscale_repeats)
            .model(value.model.clone())
            .diffusion_model(value.diffusion_model.clone())
            .llm(value.llm.clone())
            .llm_vision(value.llm_vision.clone())
            .clip_l(value.clip_l.clone())
            .clip_g(value.clip_g.clone())
            .clip_vision(value.clip_vision.clone())
            .t5xxl(value.t5xxl.clone())
            .vae(value.vae.clone())
            .audio_vae(value.audio_vae.clone())
            .taesd(value.taesd.clone())
            .control_net(value.control_net.clone())
            .embeddings(&value.embeddings.0)
            .embeddings_connectors(value.embeddings_connectors.clone())
            .photo_maker(value.photo_maker.clone())
            .pm_id_embed_path(value.pm_id_embed_path.clone())
            .weight_type(value.weight_type)
            .high_noise_diffusion_model(value.high_noise_diffusion_model.clone())
            .vae_tiling(value.vae_tiling)
            .vae_tile_size(value.vae_tile_size)
            .vae_relative_tile_size(value.vae_relative_tile_size)
            .vae_tile_overlap(value.vae_tile_overlap)
            .vae_temporal_tiling(value.vae_temporal_tiling)
            .vae_extra_tiling_args(value.vae_extra_tiling_args.clone())
            .rng(value.rng)
            .sampler_rng_type(value.rng)
            .scheduler(value.scheduler)
            .sigmas(value.sigmas.clone())
            .prediction(value.prediction)
            .vae_on_cpu(value.vae_on_cpu)
            .clip_on_cpu(value.clip_on_cpu)
            .control_net(value.control_net.clone())
            .control_net_cpu(value.control_net_cpu)
            .flash_attention(value.flash_attention)
            .chroma_disable_dit_mask(value.chroma_disable_dit_mask)
            .chroma_enable_t5_mask(value.chroma_enable_t5_mask)
            .chroma_t5_mask_pad(value.chroma_t5_mask_pad)
            .diffusion_conv_direct(value.diffusion_conv_direct)
            .vae_conv_direct(value.vae_conv_direct)
            .force_sdxl_vae_conv_scale(value.force_sdxl_vae_conv_scale)
            .flow_shift(value.flow_shift)
            .timestep_shift(value.timestep_shift)
            .taesd_preview_only(value.taesd_preview_only)
            .lora_apply_mode(value.lora_apply_mode)
            .circular(value.circular)
            .circular_x(value.circular_x)
            .circular_y(value.circular_y)
            .use_qwen_image_zero_cond_true(value.use_qwen_image_zero_cond_true)
            .hires_params(
                value.hires_params.0,
                value.hires_params.1.clone(),
                hires_path.as_deref(),
            )
            .extra_sample_params(value.extra_sample_params.clone())
            .backend(value.backend.clone())
            .params_backend(value.params_backend.clone());

        builder.lora_models_internal(value.lora_models.clone());

        if let Some(model) = &value.upscale_model {
            builder.upscale_model(model.clone());
        }
        builder
    }
}

#[derive(Builder, Debug)]
#[builder(setter(into, strip_option), build_fn(validate = "Self::validate"))]
/// Config struct common to all diffusion methods
pub struct Config {
    /// Path to PHOTOMAKER input id images dir
    #[builder(default = "Default::default()")]
    pm_id_images_dir: CLibPath,

    /// Path to the input image, required by img2img
    #[builder(default = "Default::default()")]
    init_img: InputImage,

    /// Path to the image used as a mask for img2img
    #[builder(default = "Default::default()")]
    mask_img: InputImage,

    /// Path to image condition, control net
    #[builder(default = "Default::default()")]
    control_image: CLibPath,

    /// Paths to reference images for in-context conditioning (e.g. for Flux2)
    #[builder(default = "Default::default()", setter(custom))]
    ref_images: Vec<InputImage>,

    /// Path to write result image to (default: ./output.png)
    #[builder(default = "PathBuf::from(\"./output.png\")")]
    output: PathBuf,

    /// Path to write result image to (default: ./output.png)
    #[builder(default = "PathBuf::from(\"./preview_output.png\")")]
    preview_output: PathBuf,

    /// Preview method
    #[builder(default = "PreviewType::PREVIEW_NONE")]
    preview_mode: PreviewType,

    /// Enables previewing noisy inputs of the models rather than the denoised outputs
    #[builder(default = "false")]
    preview_noisy: bool,

    /// Interval in denoising steps between consecutive updates of the image preview file (default is 1, meaning updating at every step)
    #[builder(default = "1")]
    preview_interval: i32,

    /// The prompt to render
    prompt: String,

    /// The negative prompt (default: "")
    #[builder(default = "\"\".into()")]
    negative_prompt: CLibString,

    /// Unconditional guidance scale (default: 7.0)
    #[builder(default = "7.0")]
    cfg_scale: f32,

    /// Distilled guidance scale for models with guidance input (default: 3.5)
    #[builder(default = "3.5")]
    guidance: f32,

    /// Strength for noising/unnoising (default: 0.75)
    #[builder(default = "0.75")]
    strength: f32,

    /// Strength for keeping input identity (default: 20%)
    #[builder(default = "20.0")]
    pm_style_strength: f32,

    /// Strength to apply Control Net (default: 0.9)
    /// 1.0 corresponds to full destruction of information in init
    #[builder(default = "0.9")]
    control_strength: f32,

    /// Image height, in pixel space (default: 512)
    #[builder(default = "512")]
    height: i32,

    /// Image width, in pixel space (default: 512)
    #[builder(default = "512")]
    width: i32,

    /// Sampling-method (default: [SampleMethod::SAMPLE_METHOD_COUNT]).
    /// [SampleMethod::EULER_SAMPLE_METHOD] will be used for flux, sd3, wan, qwen_image.
    /// Otherwise [SampleMethod::EULER_A_SAMPLE_METHOD] is used.
    #[builder(default = "SampleMethod::SAMPLE_METHOD_COUNT")]
    sampling_method: SampleMethod,

    /// eta in DDIM, only for DDIM/TCD/res_multistep/res_2s (default: 0)
    #[builder(default = "0.")]
    eta: f32,

    /// Number of sample steps (default: 20)
    #[builder(default = "20")]
    steps: i32,

    /// RNG seed (default: -1, use random seed for < 0)
    #[builder(default = "-1")]
    seed: i64,

    /// Number of images to generate (default: 1)
    #[builder(default = "1")]
    batch_count: i32,

    /// Ignore last layers of CLIP network; 1 ignores none, 2 ignores one layer (default: -1)
    /// <= 0 represents unspecified, will be 1 for SD1.x, 2 for SD2.x
    #[builder(default = "ClipSkip::Unspecified")]
    clip_skip: ClipSkip,

    /// Apply canny preprocessor (edge detection) (default: false)
    #[builder(default = "false")]
    canny: bool,

    /// skip layer guidance (SLG) scale, only for DiT models: (default: 0)
    /// 0 means disabled, a value of 2.5 is nice for sd3.5 medium
    #[builder(default = "0.")]
    slg_scale: f32,

    /// Layers to skip for SLG steps: (default: \[7,8,9\])
    #[builder(default = "vec![7, 8, 9]")]
    skip_layer: Vec<i32>,

    /// SLG enabling point: (default: 0.01)
    #[builder(default = "0.01")]
    skip_layer_start: f32,

    /// SLG disabling point: (default: 0.2)
    #[builder(default = "0.2")]
    skip_layer_end: f32,

    /// Disable auto resize of ref images
    #[builder(default = "false")]
    disable_auto_resize_ref_image: bool,

    #[builder(default = "Self::cache_init()", private)]
    cache: (sd_cache_params_t, Option<CLibString>),
}

#[derive(Builder, Debug, Clone)]
#[builder(
    setter(into, strip_option),
    build_fn(error = "ConfigBuilderError", validate = "Self::validate")
)]
/// Config struct for in-memory video generation
pub struct VideoConfig {
    /// Path to the input image, used as a starting frame
    #[builder(default = "Default::default()")]
    init_img: InputImage,

    /// Path to the ending image, required by some video models
    #[builder(default = "Default::default()")]
    end_img: InputImage,

    /// Paths to control video frames
    #[builder(default = "Default::default()", setter(custom))]
    control_frames: Vec<InputImage>,

    /// Preview method
    #[builder(default = "PreviewType::PREVIEW_NONE")]
    preview_mode: PreviewType,

    /// Enables previewing noisy inputs of the models rather than the denoised outputs
    #[builder(default = "false")]
    preview_noisy: bool,

    /// Interval in denoising steps between consecutive preview updates
    #[builder(default = "1")]
    preview_interval: i32,

    /// The prompt to render
    prompt: String,

    /// The negative prompt (default: "")
    #[builder(default = "\"\".into()")]
    negative_prompt: CLibString,

    /// Unconditional guidance scale (default: 7.0)
    #[builder(default = "7.0")]
    cfg_scale: f32,

    /// Distilled guidance scale for models with guidance input (default: 3.5)
    #[builder(default = "3.5")]
    guidance: f32,

    /// Strength for noising/unnoising (default: 0.75)
    #[builder(default = "0.75")]
    strength: f32,

    /// Image height, in pixel space (default: 512)
    #[builder(default = "512")]
    height: i32,

    /// Image width, in pixel space (default: 512)
    #[builder(default = "512")]
    width: i32,

    /// Sampling method for the main denoising phase
    #[builder(default = "SampleMethod::SAMPLE_METHOD_COUNT")]
    sampling_method: SampleMethod,

    /// eta in DDIM, only for DDIM/TCD/res_multistep/res_2s (default: 0)
    #[builder(default = "0.")]
    eta: f32,

    /// Number of sample steps (default: 20)
    #[builder(default = "20")]
    steps: i32,

    /// RNG seed (default: -1, use random seed)
    #[builder(default = "-1")]
    seed: i64,

    /// Number of frames requested from the backend (default: 6)
    #[builder(default = "6")]
    video_frames: i32,

    /// Output frames per second metadata/conditioning for video generation (default: 0)
    #[builder(default = "0")]
    fps: i32,

    /// Ignore last layers of CLIP network; 1 ignores none, 2 ignores one layer (default: -1)
    #[builder(default = "ClipSkip::Unspecified")]
    clip_skip: ClipSkip,

    /// skip layer guidance (SLG) scale, only for DiT models: (default: 0)
    #[builder(default = "0.")]
    slg_scale: f32,

    /// Layers to skip for SLG steps: (default: [7,8,9])
    #[builder(default = "vec![7, 8, 9]")]
    skip_layer: Vec<i32>,

    /// SLG enabling point: (default: 0.01)
    #[builder(default = "0.01")]
    skip_layer_start: f32,

    /// SLG disabling point: (default: 0.2)
    #[builder(default = "0.2")]
    skip_layer_end: f32,

    /// Number of high-noise sample steps (default: -1)
    #[builder(default = "-1")]
    high_noise_steps: i32,

    /// Sampling method for the high-noise phase
    #[builder(default = "SampleMethod::SAMPLE_METHOD_COUNT")]
    high_noise_sampling_method: SampleMethod,

    /// eta for the high-noise phase
    #[builder(default = "0.")]
    high_noise_eta: f32,

    /// Extra parameters for high-noise sampling, in backend-specific string format
    #[builder(default = "CLibString::default()")]
    high_noise_extra_sample_params: CLibString,

    /// Unconditional guidance scale for the high-noise phase (default: 7.0)
    #[builder(default = "7.0")]
    high_noise_cfg_scale: f32,

    /// Distilled guidance scale for the high-noise phase (default: 3.5)
    #[builder(default = "3.5")]
    high_noise_guidance: f32,

    /// skip layer guidance (SLG) scale for the high-noise phase
    #[builder(default = "0.")]
    high_noise_slg_scale: f32,

    /// Layers to skip for high-noise SLG steps: (default: [7,8,9])
    #[builder(default = "vec![7, 8, 9]")]
    high_noise_skip_layer: Vec<i32>,

    /// High-noise SLG enabling point: (default: 0.01)
    #[builder(default = "0.01")]
    high_noise_skip_layer_start: f32,

    /// High-noise SLG disabling point: (default: 0.2)
    #[builder(default = "0.2")]
    high_noise_skip_layer_end: f32,

    /// timestep boundary for Wan2.2 MoE model. (default: 0.875)
    #[builder(default = "0.875")]
    moe_boundary: f32,

    /// Wan VACE strength (default: 1.0)
    #[builder(default = "1.0")]
    vace_strength: f32,

    /// Video-specific hires fix parameters and upscaler model.
    #[builder(default = "ModelConfigBuilder::hires_init()", setter(custom))]
    hires_params: (Upscaler, HiresParams, Option<CLibPath>),

    #[builder(default = "ConfigBuilder::cache_init()", private)]
    cache: (sd_cache_params_t, Option<CLibString>),
}

impl ConfigBuilder {
    fn validate(&self) -> Result<(), ConfigBuilderError> {
        self.validate_output_dir()
    }

    fn validate_output_dir(&self) -> Result<(), ConfigBuilderError> {
        let is_dir = self.output.as_ref().is_some_and(|val| val.is_dir());
        let multiple_items = self.batch_count.as_ref().is_some_and(|val| *val > 1);
        if is_dir == multiple_items {
            Ok(())
        } else {
            Err(ConfigBuilderError::ValidationError(
                "When batch_count > 1, output should point to folder and vice versa".to_owned(),
            ))
        }
    }

    fn cache_init() -> (sd_cache_params_t, Option<CLibString>) {
        (
            sd_cache_params_t {
                mode: sd_cache_mode_t::SD_CACHE_DISABLED,
                reuse_threshold: 1.0,
                start_percent: 0.15,
                end_percent: 0.95,
                error_decay_rate: 1.0,
                use_relative_threshold: true,
                reset_error_on_compute: true,
                Fn_compute_blocks: 8,
                Bn_compute_blocks: 0,
                residual_diff_threshold: 0.08,
                max_warmup_steps: 8,
                max_cached_steps: -1,
                max_continuous_cached_steps: -1,
                taylorseer_n_derivatives: 1,
                taylorseer_skip_interval: 1,
                scm_mask: null(),
                scm_policy_dynamic: true,
                spectrum_w: 0.4,
                spectrum_m: 3,
                spectrum_lam: 1.0,
                spectrum_window_size: 2,
                spectrum_flex_window: 0.5,
                spectrum_warmup_steps: 4,
                spectrum_stop_percent: 0.9,
            },
            None,
        )
    }

    pub fn no_caching(&mut self) -> &mut Self {
        let mut cache = Self::cache_init();
        cache.0.mode = sd_cache_mode_t::SD_CACHE_DISABLED;
        self.cache = Some(cache);
        self
    }

    pub fn ref_images<I>(&mut self, ref_images: Vec<I>) -> &mut Self
    where
        I: Into<InputImage>,
    {
        self.ref_images = Some(ref_images.into_iter().map(Into::into).collect());
        self
    }

    pub fn spectrum_caching(&mut self, params: SpectrumCacheParams) -> &mut Self {
        let (mut cache, mask) = Self::cache_init();
        cache.mode = sd_cache_mode_t::SD_CACHE_SPECTRUM;
        cache.spectrum_w = params.w;
        cache.spectrum_m = params.m;
        cache.spectrum_lam = params.lam;
        cache.spectrum_window_size = params.window;
        cache.spectrum_flex_window = params.flex;
        cache.spectrum_warmup_steps = params.warmup;
        cache.spectrum_stop_percent = params.stop;
        self.cache = Some((cache, mask));
        self
    }

    pub fn ucache_caching(&mut self, params: UCacheParams) -> &mut Self {
        let (mut cache, mask) = Self::cache_init();
        cache.mode = sd_cache_mode_t::SD_CACHE_UCACHE;
        cache.reuse_threshold = params.threshold;
        cache.start_percent = params.start;
        cache.end_percent = params.end;
        cache.error_decay_rate = params.decay;
        cache.use_relative_threshold = params.relative;
        cache.reset_error_on_compute = params.reset;
        self.cache = Some((cache, mask));
        self
    }

    pub fn easy_cache_caching(&mut self, params: EasyCacheParams) -> &mut Self {
        let (mut cache, mask) = Self::cache_init();
        cache.mode = sd_cache_mode_t::SD_CACHE_EASYCACHE;
        cache.reuse_threshold = params.threshold;
        cache.start_percent = params.start;
        cache.end_percent = params.end;
        self.cache = Some((cache, mask));
        self
    }

    pub fn db_cache_caching(&mut self, params: DbCacheParams) -> &mut Self {
        let (mut cache, _) = Self::cache_init();
        cache.mode = sd_cache_mode_t::SD_CACHE_DBCACHE;
        cache.Fn_compute_blocks = params.fn_blocks;
        cache.Bn_compute_blocks = params.bn_blocks;
        cache.residual_diff_threshold = params.threshold;
        cache.max_warmup_steps = params.warmup;
        cache.scm_policy_dynamic = match params.scm_policy_dynamic {
            ScmPolicy::Static => false,
            ScmPolicy::Dynamic => true,
        };
        self.cache = Some((cache, Some(params.scm_mask)));
        self
    }

    pub fn taylor_seer_caching(&mut self) -> &mut Self {
        let (mut cache, mask) = Self::cache_init();
        cache.mode = sd_cache_mode_t::SD_CACHE_TAYLORSEER;
        self.cache = Some((cache, mask));
        self
    }

    pub fn cache_dit_caching(&mut self, params: DbCacheParams) -> &mut Self {
        self.db_cache_caching(params).cache.as_mut().unwrap().0.mode =
            sd_cache_mode_t::SD_CACHE_CACHE_DIT;
        self
    }
}

impl VideoConfigBuilder {
    fn validate(&self) -> Result<(), ConfigBuilderError> {
        self.validate_video_frames()?;
        self.validate_image_paths()
    }

    fn validate_video_frames(&self) -> Result<(), ConfigBuilderError> {
        if self
            .video_frames
            .as_ref()
            .map_or(true, |frames| *frames > 0)
        {
            Ok(())
        } else {
            Err(ConfigBuilderError::ValidationError(
                "video_frames must be greater than 0".to_owned(),
            ))
        }
    }

    fn validate_image_paths(&self) -> Result<(), ConfigBuilderError> {
        for (label, maybe_path) in [
            ("init_img", self.init_img.as_ref()),
            ("end_img", self.end_img.as_ref()),
        ] {
            if let Some(image) = maybe_path
                && image.is_present()
            {
                validate_readable_image(image, label)?;
            }
        }

        if let Some(control_frames) = self.control_frames.as_ref() {
            for image in control_frames {
                validate_readable_image(image, "control_frames")?;
            }
        }

        Ok(())
    }

    pub fn control_frames<I>(&mut self, control_frames: Vec<I>) -> &mut Self
    where
        I: Into<InputImage>,
    {
        self.control_frames = Some(control_frames.into_iter().map(Into::into).collect());
        self
    }

    pub fn hires_params(
        &mut self,
        upscaler: Upscaler,
        params: HiresParams,
        custom_model: Option<&Path>,
    ) -> &mut Self {
        if upscaler == Upscaler::SD_HIRES_UPSCALER_COUNT
            || (upscaler == Upscaler::SD_HIRES_UPSCALER_MODEL && custom_model.is_none())
        {
            panic!("Invalid combination for {upscaler:?} and {custom_model:?}")
        }
        self.hires_params = Some((upscaler, params, custom_model.map(Into::into)));
        self
    }

    pub fn video_hires_params(
        &mut self,
        upscaler: Upscaler,
        params: HiresParams,
        custom_model: Option<&Path>,
    ) -> &mut Self {
        self.hires_params(upscaler, params, custom_model)
    }

    pub fn no_caching(&mut self) -> &mut Self {
        let mut cache = ConfigBuilder::cache_init();
        cache.0.mode = sd_cache_mode_t::SD_CACHE_DISABLED;
        self.cache = Some(cache);
        self
    }

    pub fn spectrum_caching(&mut self, params: SpectrumCacheParams) -> &mut Self {
        let (mut cache, mask) = ConfigBuilder::cache_init();
        cache.mode = sd_cache_mode_t::SD_CACHE_SPECTRUM;
        cache.spectrum_w = params.w;
        cache.spectrum_m = params.m;
        cache.spectrum_lam = params.lam;
        cache.spectrum_window_size = params.window;
        cache.spectrum_flex_window = params.flex;
        cache.spectrum_warmup_steps = params.warmup;
        cache.spectrum_stop_percent = params.stop;
        self.cache = Some((cache, mask));
        self
    }

    pub fn ucache_caching(&mut self, params: UCacheParams) -> &mut Self {
        let (mut cache, mask) = ConfigBuilder::cache_init();
        cache.mode = sd_cache_mode_t::SD_CACHE_UCACHE;
        cache.reuse_threshold = params.threshold;
        cache.start_percent = params.start;
        cache.end_percent = params.end;
        cache.error_decay_rate = params.decay;
        cache.use_relative_threshold = params.relative;
        cache.reset_error_on_compute = params.reset;
        self.cache = Some((cache, mask));
        self
    }

    pub fn easy_cache_caching(&mut self, params: EasyCacheParams) -> &mut Self {
        let (mut cache, mask) = ConfigBuilder::cache_init();
        cache.mode = sd_cache_mode_t::SD_CACHE_EASYCACHE;
        cache.reuse_threshold = params.threshold;
        cache.start_percent = params.start;
        cache.end_percent = params.end;
        self.cache = Some((cache, mask));
        self
    }

    pub fn db_cache_caching(&mut self, params: DbCacheParams) -> &mut Self {
        let (mut cache, _) = ConfigBuilder::cache_init();
        cache.mode = sd_cache_mode_t::SD_CACHE_DBCACHE;
        cache.Fn_compute_blocks = params.fn_blocks;
        cache.Bn_compute_blocks = params.bn_blocks;
        cache.residual_diff_threshold = params.threshold;
        cache.max_warmup_steps = params.warmup;
        cache.scm_policy_dynamic = match params.scm_policy_dynamic {
            ScmPolicy::Static => false,
            ScmPolicy::Dynamic => true,
        };
        self.cache = Some((cache, Some(params.scm_mask)));
        self
    }

    pub fn taylor_seer_caching(&mut self) -> &mut Self {
        let (mut cache, mask) = ConfigBuilder::cache_init();
        cache.mode = sd_cache_mode_t::SD_CACHE_TAYLORSEER;
        self.cache = Some((cache, mask));
        self
    }

    pub fn cache_dit_caching(&mut self, params: DbCacheParams) -> &mut Self {
        self.db_cache_caching(params).cache.as_mut().unwrap().0.mode =
            sd_cache_mode_t::SD_CACHE_CACHE_DIT;
        self
    }
}

impl From<&Config> for ConfigBuilder {
    fn from(value: &Config) -> Self {
        let mut builder = ConfigBuilder::default();
        builder
            .pm_id_images_dir(value.pm_id_images_dir.clone())
            .init_img(value.init_img.clone())
            .mask_img(value.mask_img.clone())
            .control_image(value.control_image.clone())
            .ref_images(value.ref_images.clone())
            .output(value.output.clone())
            .prompt(value.prompt.clone())
            .negative_prompt(value.negative_prompt.clone())
            .cfg_scale(value.cfg_scale)
            .strength(value.strength)
            .pm_style_strength(value.pm_style_strength)
            .control_strength(value.control_strength)
            .height(value.height)
            .width(value.width)
            .sampling_method(value.sampling_method)
            .steps(value.steps)
            .seed(value.seed)
            .batch_count(value.batch_count)
            .clip_skip(value.clip_skip)
            .slg_scale(value.slg_scale)
            .skip_layer(value.skip_layer.clone())
            .skip_layer_start(value.skip_layer_start)
            .skip_layer_end(value.skip_layer_end)
            .canny(value.canny)
            .disable_auto_resize_ref_image(value.disable_auto_resize_ref_image)
            .preview_output(value.preview_output.clone())
            .preview_mode(value.preview_mode)
            .preview_noisy(value.preview_noisy)
            .preview_interval(value.preview_interval)
            .cache(value.cache.clone());
        builder
    }
}

impl From<VideoConfig> for VideoConfigBuilder {
    fn from(value: VideoConfig) -> Self {
        let mut builder = VideoConfigBuilder::default();
        let hires_path = value.hires_params.2.as_ref().map(PathBuf::from);
        builder
            .init_img(value.init_img)
            .end_img(value.end_img)
            .control_frames(value.control_frames)
            .prompt(value.prompt)
            .negative_prompt(value.negative_prompt)
            .cfg_scale(value.cfg_scale)
            .guidance(value.guidance)
            .strength(value.strength)
            .height(value.height)
            .width(value.width)
            .sampling_method(value.sampling_method)
            .eta(value.eta)
            .steps(value.steps)
            .seed(value.seed)
            .video_frames(value.video_frames)
            .fps(value.fps)
            .clip_skip(value.clip_skip)
            .slg_scale(value.slg_scale)
            .skip_layer(value.skip_layer)
            .skip_layer_start(value.skip_layer_start)
            .skip_layer_end(value.skip_layer_end)
            .high_noise_steps(value.high_noise_steps)
            .high_noise_sampling_method(value.high_noise_sampling_method)
            .high_noise_eta(value.high_noise_eta)
            .high_noise_extra_sample_params(value.high_noise_extra_sample_params)
            .high_noise_cfg_scale(value.high_noise_cfg_scale)
            .high_noise_guidance(value.high_noise_guidance)
            .high_noise_slg_scale(value.high_noise_slg_scale)
            .high_noise_skip_layer(value.high_noise_skip_layer)
            .high_noise_skip_layer_start(value.high_noise_skip_layer_start)
            .high_noise_skip_layer_end(value.high_noise_skip_layer_end)
            .moe_boundary(value.moe_boundary)
            .vace_strength(value.vace_strength)
            .preview_mode(value.preview_mode)
            .preview_noisy(value.preview_noisy)
            .preview_interval(value.preview_interval)
            .hires_params(
                value.hires_params.0,
                value.hires_params.1,
                hires_path.as_deref(),
            )
            .cache(value.cache.clone());
        builder
    }
}

#[derive(Debug, Clone, Default)]
struct CLibString(CString);

impl CLibString {
    fn as_ptr(&self) -> *const c_char {
        self.0.as_ptr()
    }

    fn as_optional_ptr(&self) -> *const c_char {
        if self.0.as_bytes().is_empty() {
            null()
        } else {
            self.as_ptr()
        }
    }
}

impl From<&str> for CLibString {
    fn from(value: &str) -> Self {
        Self(CString::new(value).unwrap())
    }
}

impl From<String> for CLibString {
    fn from(value: String) -> Self {
        Self(CString::new(value).unwrap())
    }
}

#[derive(Debug, Clone, Default)]
struct CLibPath(CString);

impl CLibPath {
    fn as_ptr(&self) -> *const c_char {
        self.0.as_ptr()
    }
}

impl From<PathBuf> for CLibPath {
    fn from(value: PathBuf) -> Self {
        Self(CString::new(value.to_str().unwrap_or_default()).unwrap())
    }
}

impl From<&Path> for CLibPath {
    fn from(value: &Path) -> Self {
        Self(CString::new(value.to_str().unwrap_or_default()).unwrap())
    }
}

impl From<&CLibPath> for PathBuf {
    fn from(value: &CLibPath) -> Self {
        PathBuf::from(value.0.to_str().unwrap())
    }
}

fn output_files(path: &Path, batch_size: i32) -> Vec<PathBuf> {
    let date = Local::now().format("%Y.%m.%d-%H.%M.%S");
    if batch_size == 1 {
        vec![path.into()]
    } else {
        (1..=batch_size)
            .map(|id| path.join(format!("output_{date}_{id}.png")))
            .collect()
    }
}

fn validate_readable_image(image: &InputImage, label: &str) -> Result<(), ConfigBuilderError> {
    if let InputImage::Path(path) = image {
        if !path.exists() {
            return Err(ConfigBuilderError::ValidationError(format!(
                "{label} path does not exist: {}",
                path.display()
            )));
        }

        image::open(path).map_err(|err| {
            ConfigBuilderError::ValidationError(format!(
                "{label} path is not a readable image ({}): {err}",
                path.display()
            ))
        })?;
    }

    Ok(())
}

fn load_rgb_image(source: &InputImage) -> Result<RgbImage, DiffusionError> {
    match source {
        InputImage::Path(path) => Ok(image::open(path)?.to_rgb8()),
        InputImage::Image(image) => Ok(image.clone()),
    }
}

fn make_guidance(
    cfg_scale: f32,
    guidance: f32,
    slg_scale: f32,
    layers: &mut [i32],
    layer_start: f32,
    layer_end: f32,
) -> sd_guidance_params_t {
    sd_guidance_params_t {
        txt_cfg: cfg_scale,
        img_cfg: cfg_scale,
        distilled_guidance: guidance,
        slg: sd_slg_params_t {
            layers: layers.as_mut_ptr(),
            layer_count: layers.len(),
            layer_start,
            layer_end,
            scale: slg_scale,
        },
    }
}

fn make_hires_params(
    upscaler: Upscaler,
    params: &HiresParams,
    custom_model: Option<&CLibPath>,
    custom_sigmas: &mut Vec<f32>,
) -> sd_hires_params_t {
    sd_hires_params_t {
        enabled: upscaler != Upscaler::SD_HIRES_UPSCALER_NONE,
        upscaler,
        model_path: custom_model.map_or(null(), CLibPath::as_ptr),
        scale: params.scale,
        target_width: params.width,
        target_height: params.height,
        steps: params.steps,
        denoising_strength: params.denoising_strength,
        upscale_tile_size: params.upscale_tile_size,
        custom_sigmas: custom_sigmas.as_mut_ptr(),
        custom_sigmas_count: custom_sigmas.len() as i32,
    }
}

fn image_from_sd_image(img: sd_image_t) -> Option<RgbImage> {
    let len = (img.width * img.height * img.channel) as usize;
    let buffer = unsafe { slice::from_raw_parts(img.data, len).to_vec() };
    ImageBuffer::from_raw(img.width, img.height, buffer).map(RgbImage::from)
}

fn images_from_sd_frames(frames: *mut sd_image_t, frame_count: i32) -> Vec<RgbImage> {
    if frames.is_null() || frame_count <= 0 {
        return Vec::new();
    }

    unsafe { slice::from_raw_parts(frames, frame_count as usize) }
        .iter()
        .copied()
        .filter(|img| !img.data.is_null())
        .filter_map(image_from_sd_image)
        .collect()
}

fn audio_from_sd_audio(audio: *const sd_audio_t) -> Option<AudioBuffer> {
    if audio.is_null() {
        return None;
    }

    let audio = unsafe { &*audio };
    if audio.data.is_null() || audio.sample_count == 0 || audio.channels == 0 {
        return None;
    }

    let sample_count = usize::try_from(audio.sample_count).ok()?;
    let channels = usize::try_from(audio.channels).ok()?;
    let len = sample_count.checked_mul(channels)?;
    let samples = unsafe { slice::from_raw_parts(audio.data, len).to_vec() };

    Some(AudioBuffer {
        sample_rate: audio.sample_rate,
        channels: audio.channels,
        samples,
    })
}

fn final_video_progress(final_frames: Vec<RgbImage>, audio: *const sd_audio_t) -> ExtProgress {
    match audio_from_sd_audio(audio) {
        Some(audio) => ExtProgress::FinalFramesWithAudio(final_frames, audio),
        None => ExtProgress::FinalFrames(final_frames),
    }
}

unsafe fn upscale(
    upscale_repeats: i32,
    upscaler_ctx: Option<*mut upscaler_ctx_t>,
    data: sd_image_t,
) -> Result<sd_image_t, DiffusionError> {
    unsafe {
        match upscaler_ctx {
            Some(upscaler_ctx) => {
                let upscale_factor = 4; // unused for RealESRGAN_x4plus_anime_6B.pth
                let mut current_image = data;
                for _ in 0..upscale_repeats {
                    let upscaled_image =
                        diffusion_rs_sys::upscale(upscaler_ctx, current_image, upscale_factor);

                    if upscaled_image.data.is_null() {
                        return Err(DiffusionError::Upscaler);
                    }

                    free(current_image.data as *mut c_void);
                    current_image = upscaled_image;
                }
                Ok(current_image)
            }
            None => Ok(data),
        }
    }
}

/// Generate an image and receive progress, preview images and final images
/// via callback.
/// Note: All images are exclusively delivered in-memory via the callback, and the file output specified in the config is ignored when using this function.
pub fn gen_img_with_ext_progress(
    config: &Config,
    model_config: &mut ModelConfig,
    sender: Sender<ExtProgress>,
) -> Result<(), DiffusionError> {
    gen_img_maybe_progress(config, model_config, None, Some(sender))
}

/// Generate a video and stream previews/final frames in-memory.
///
/// Callers should expect `ExtProgress::PreviewFrames` preview events and a final
/// `ExtProgress::FinalFrames` or `ExtProgress::FinalFramesWithAudio` event.
/// `stable-diffusion.cpp` may normalize the final frame count to `4n+1`, so the
/// returned frame count can differ from the requested `video_frames`.
pub fn gen_video_with_ext_progress(
    config: &VideoConfig,
    model_config: &mut ModelConfig,
    sender: Sender<ExtProgress>,
) -> Result<(), DiffusionError> {
    gen_video_maybe_progress(config, model_config, None, Some(sender))
}

/// Generate an image and receive update via queue
pub fn gen_img_with_progress(
    config: &Config,
    model_config: &mut ModelConfig,
    sender: Sender<Progress>,
) -> Result<(), DiffusionError> {
    gen_img_maybe_progress(config, model_config, Some(sender), None)
}

/// Generate a video while only receiving step-based progress updates.
pub fn gen_video_with_progress(
    config: &VideoConfig,
    model_config: &mut ModelConfig,
    sender: Sender<Progress>,
) -> Result<(), DiffusionError> {
    gen_video_maybe_progress(config, model_config, Some(sender), None)
}

/// Generate an image
pub fn gen_img(config: &Config, model_config: &mut ModelConfig) -> Result<(), DiffusionError> {
    gen_img_maybe_progress(config, model_config, None, None)
}

fn gen_img_maybe_progress(
    config: &Config,
    model_config: &mut ModelConfig,
    mut sender: Option<Sender<Progress>>,
    mut ext_sender: Option<Sender<ExtProgress>>,
) -> Result<(), DiffusionError> {
    let prompt: CLibString = CLibString::from(config.prompt.as_str());
    let files = output_files(&config.output, config.batch_count);
    unsafe {
        struct ScopeGuard;
        impl Drop for ScopeGuard {
            fn drop(&mut self) {
                unsafe {
                    sd_set_progress_callback(None, null_mut());
                    sd_set_preview_callback(
                        None,
                        PreviewType::PREVIEW_NONE,
                        0,
                        false,
                        false,
                        null_mut(),
                    );
                }
            }
        }
        let _guard = ScopeGuard;
        let has_init_image = config.init_img.is_present();
        let has_mask_image = config.mask_img.is_present();
        let has_ref_images = !config.ref_images.is_empty();

        let is_decode_only = !has_init_image && !has_ref_images;
        let sd_ctx = model_config.diffusion_ctx(is_decode_only);
        let upscaler_ctx = model_config.upscaler_ctx();

        let mut init_image = sd_image_t {
            width: 0,
            height: 0,
            channel: 3,
            data: std::ptr::null_mut(),
        };
        let mut mask_image = sd_image_t {
            width: config.width as u32,
            height: config.height as u32,
            channel: 1,
            data: null_mut(),
        };
        let mut layers = config.skip_layer.clone();
        let guidance = sd_guidance_params_t {
            txt_cfg: config.cfg_scale,
            img_cfg: config.cfg_scale,
            distilled_guidance: config.guidance,
            slg: sd_slg_params_t {
                layers: layers.as_mut_ptr(),
                layer_count: config.skip_layer.len(),
                layer_start: config.skip_layer_start,
                layer_end: config.skip_layer_end,
                scale: config.slg_scale,
            },
        };
        let scheduler = if model_config.scheduler == Scheduler::SCHEDULER_COUNT {
            sd_get_default_scheduler(sd_ctx, config.sampling_method)
        } else {
            model_config.scheduler
        };
        let sample_method = if config.sampling_method == SampleMethod::SAMPLE_METHOD_COUNT {
            sd_get_default_sample_method(sd_ctx)
        } else {
            config.sampling_method
        };
        let sample_params = sd_sample_params_t {
            guidance,
            sample_method,
            sample_steps: config.steps,
            eta: config.eta,
            scheduler,
            shifted_timestep: model_config.timestep_shift,
            custom_sigmas: model_config.sigmas.as_mut_ptr(),
            custom_sigmas_count: model_config.sigmas.len() as i32,
            flow_shift: model_config.flow_shift,
            extra_sample_args: model_config.extra_sample_params.as_optional_ptr(),
        };
        let control_image = sd_image_t {
            width: 0,
            height: 0,
            channel: 3,
            data: null_mut(),
        };
        let vae_tiling_params = sd_tiling_params_t {
            enabled: model_config.vae_tiling,
            temporal_tiling: model_config.vae_temporal_tiling,
            tile_size_x: model_config.vae_tile_size.0,
            tile_size_y: model_config.vae_tile_size.1,
            target_overlap: model_config.vae_tile_overlap,
            rel_size_x: model_config.vae_relative_tile_size.0,
            rel_size_y: model_config.vae_relative_tile_size.1,
            extra_tiling_args: model_config.vae_extra_tiling_args.as_optional_ptr(),
        };
        let pm_params = sd_pm_params_t {
            id_images: null_mut(),
            id_images_count: 0,
            id_embed_path: model_config.pm_id_embed_path.as_ptr(),
            style_strength: config.pm_style_strength,
        };

        // Declare the buffer in the function scope so it outlives the match block
        let mut image_buffer: Vec<u8> = Vec::new();
        let mut mask_buffer: Vec<u8> = Vec::new();

        if has_init_image {
            let img = load_rgb_image(&config.init_img)?;
            let (width, height) = img.dimensions();
            image_buffer = img.into_raw();

            init_image = sd_image_t {
                width,
                height,
                channel: 3,
                data: image_buffer.as_mut_ptr(),
            }
        }

        if has_mask_image {
            let img = load_rgb_image(&config.mask_img)?;
            let (width, height) = img.dimensions();
            // Masks have to have single channel luminosity information only
            let gray = image::DynamicImage::ImageRgb8(img).to_luma8();
            mask_buffer = gray.into_raw();

            mask_image = sd_image_t {
                width,
                height,
                channel: 1,
                data: mask_buffer.as_mut_ptr(),
            }
        }

        // stable-diffusion.cpp assumes that a mask is also included with the img2img flow
        // if a mask is not provided, we use a flat white mask meaning all of the image is in scope
        // otherwise generate_image throws a sigsegv when it tries to assign the mask
        if !image_buffer.is_empty() && mask_buffer.is_empty() {
            let img: ImageBuffer<image::Luma<u8>, Vec<u8>> =
                ImageBuffer::from_pixel(init_image.width, init_image.height, image::Luma([255]));
            mask_buffer = img.into_raw();
            mask_image = sd_image_t {
                width: init_image.width,
                height: init_image.height,
                channel: 1,
                data: mask_buffer.as_mut_ptr(),
            }
        }

        let mut ref_image_list = Vec::new();
        let mut ref_pixel_storage = Vec::new();
        for ref_image in &config.ref_images {
            let img = load_rgb_image(ref_image)?;
            let (width, height) = img.dimensions();
            let image_data = img.into_raw();

            ref_pixel_storage.push(image_data);
            let storage_ref = ref_pixel_storage.last_mut().unwrap();
            ref_image_list.push(sd_image_t {
                width,
                height,
                channel: 3,
                data: storage_ref.as_mut_ptr(),
            });
        }

        let num_ref_images = ref_image_list.len();
        let ref_image_ptr = if num_ref_images > 0 {
            ref_image_list.as_mut_ptr()
        } else {
            null_mut()
        };

        unsafe extern "C" fn save_preview_local(
            _step: ::std::os::raw::c_int,
            _frame_count: ::std::os::raw::c_int,
            frames: *mut sd_image_t,
            _is_noisy: bool,
            data: *mut ::std::os::raw::c_void,
        ) {
            let _ = catch_unwind(|| unsafe {
                let path = &*data.cast::<PathBuf>();
                let _ = save_img(*frames, path, None);
            });
        }

        unsafe extern "C" fn send_preview_local(
            _step: ::std::os::raw::c_int,
            _frame_count: ::std::os::raw::c_int,
            frames: *mut sd_image_t,
            _is_noisy: bool,
            data: *mut ::std::os::raw::c_void,
        ) {
            let _ = catch_unwind(|| unsafe {
                let sender = &*data.cast::<Option<Sender<ExtProgress>>>();
                let Some(sender) = sender else {
                    return;
                };
                let img = *frames;

                let len = (img.width * img.height * img.channel) as usize;
                let buffer = slice::from_raw_parts(img.data, len).to_vec();
                let _ = ImageBuffer::from_raw(img.width, img.height, buffer)
                    .map(|img| RgbImage::from(img))
                    .and_then(|img| sender.send(ExtProgress::Preview(img)).ok());
            });
        }

        if config.preview_mode != PreviewType::PREVIEW_NONE {
            match ext_sender {
                Some(_) => {
                    sd_set_preview_callback(
                        Some(send_preview_local),
                        config.preview_mode,
                        config.preview_interval,
                        !config.preview_noisy,
                        config.preview_noisy,
                        &mut ext_sender as *mut _ as *mut c_void,
                    );
                }
                None => {
                    let data = &config.preview_output as *const PathBuf;

                    sd_set_preview_callback(
                        Some(save_preview_local),
                        config.preview_mode,
                        config.preview_interval,
                        !config.preview_noisy,
                        config.preview_noisy,
                        data as *mut c_void,
                    );
                }
            }
        }

        if sender.is_some() {
            unsafe extern "C" fn progress_callback(
                step: ::std::os::raw::c_int,
                steps: ::std::os::raw::c_int,
                time: f32,
                data: *mut ::std::os::raw::c_void,
            ) {
                let _ = catch_unwind(|| unsafe {
                    let sender = &*data.cast::<Option<Sender<Progress>>>();

                    if let Some(sender) = sender {
                        let _ = sender.send(Progress { step, steps, time });
                    }
                });
            }
            let sender_ptr: *mut c_void = &mut sender as *mut _ as *mut c_void;
            sd_set_progress_callback(Some(progress_callback), sender_ptr);
        } else if ext_sender.is_some() {
            unsafe extern "C" fn ext_progress_callback(
                step: ::std::os::raw::c_int,
                steps: ::std::os::raw::c_int,
                time: f32,
                data: *mut ::std::os::raw::c_void,
            ) {
                let _ = catch_unwind(|| unsafe {
                    let sender = &*data.cast::<Option<Sender<ExtProgress>>>();

                    if let Some(sender) = sender {
                        let _ = sender.send(ExtProgress::Progress(Progress { step, steps, time }));
                    }
                });
            }
            let sender_ptr: *mut c_void = &mut ext_sender as *mut _ as *mut c_void;
            sd_set_progress_callback(Some(ext_progress_callback), sender_ptr);
        }

        let loras: Vec<sd_lora_t> = model_config
            .lora_models
            .iter()
            .map(|(c_path, spec)| sd_lora_t {
                is_high_noise: spec.is_high_noise,
                multiplier: spec.multiplier,
                path: c_path.as_ptr(),
            })
            .collect();

        let mut cache = config.cache.0;
        if let Some(scm_mask) = &config.cache.1 {
            cache.scm_mask = scm_mask.as_ptr();
        }

        let mut hires_custom_sigmas = model_config.hires_params.1.custom_sigmas.clone();
        let hires = make_hires_params(
            model_config.hires_params.0,
            &model_config.hires_params.1,
            model_config.hires_params.2.as_ref(),
            &mut hires_custom_sigmas,
        );

        let sd_img_gen_params = sd_img_gen_params_t {
            prompt: prompt.as_ptr(),
            negative_prompt: config.negative_prompt.as_ptr(),
            clip_skip: config.clip_skip as i32,
            init_image,
            ref_images: ref_image_ptr,
            ref_images_count: num_ref_images as i32,
            increase_ref_index: false,
            mask_image,
            width: config.width,
            height: config.height,
            sample_params,
            strength: config.strength,
            seed: config.seed,
            batch_count: config.batch_count,
            control_image,
            control_strength: config.control_strength,
            pm_params,
            vae_tiling_params,
            auto_resize_ref_image: !config.disable_auto_resize_ref_image,
            cache,
            loras: loras.as_ptr(),
            lora_count: loras.len() as u32,
            hires,
        };

        let params_str = CString::from_raw(sd_img_gen_params_to_str(&sd_img_gen_params))
            .into_string()
            .unwrap();

        let slice = generate_image(sd_ctx, &sd_img_gen_params);
        let ret = {
            if slice.is_null() {
                return Err(DiffusionError::Forward);
            }
            let image_slices = slice::from_raw_parts(slice, config.batch_count as usize);
            if ext_sender.is_some() {
                for img in image_slices.iter() {
                    // img.data will be null on OOM or other generation errors,
                    // in which case we skip sending the preview and just return an error
                    if img.data.is_null() {
                        return Err(DiffusionError::Forward);
                    }
                    match upscale(model_config.upscale_repeats, upscaler_ctx, *img) {
                        Ok(img) => {
                            let len = (img.width * img.height * img.channel) as usize;
                            let buffer = slice::from_raw_parts(img.data, len).to_vec();
                            let _ = ImageBuffer::from_raw(img.width, img.height, buffer)
                                .map(|img| RgbImage::from(img))
                                .and_then(|img| {
                                    ext_sender
                                        .as_ref()
                                        .unwrap()
                                        .send(ExtProgress::FinalImage(img))
                                        .ok()
                                });
                        }
                        Err(err) => {
                            return Err(err);
                        }
                    }
                }
            } else {
                for (img, path) in image_slices.iter().zip(files) {
                    // img.data will be null on OOM or other generation errors,
                    // in which case we skip saving and just return an error
                    if img.data.is_null() {
                        return Err(DiffusionError::Forward);
                    }
                    match upscale(model_config.upscale_repeats, upscaler_ctx, *img) {
                        Ok(img) => save_img(img, &path, Some(&params_str))?,
                        Err(err) => {
                            return Err(err);
                        }
                    }
                }
            }
            Ok(())
        };
        free(slice as *mut c_void);
        ret
    }
}

fn gen_video_maybe_progress(
    config: &VideoConfig,
    model_config: &mut ModelConfig,
    mut sender: Option<Sender<Progress>>,
    mut ext_sender: Option<Sender<ExtProgress>>,
) -> Result<(), DiffusionError> {
    let prompt: CLibString = CLibString::from(config.prompt.as_str());
    unsafe {
        struct ScopeGuard;
        impl Drop for ScopeGuard {
            fn drop(&mut self) {
                unsafe {
                    sd_set_progress_callback(None, null_mut());
                    sd_set_preview_callback(
                        None,
                        PreviewType::PREVIEW_NONE,
                        0,
                        false,
                        false,
                        null_mut(),
                    );
                }
            }
        }
        let _guard = ScopeGuard;

        let has_init_image = config.init_img.is_present();
        let has_end_image = config.end_img.is_present();
        let has_control_frames = !config.control_frames.is_empty();

        let is_decode_only = !(has_init_image || has_end_image || has_control_frames);
        let sd_ctx = model_config.diffusion_ctx(is_decode_only);

        let mut init_image = sd_image_t {
            width: 0,
            height: 0,
            channel: 3,
            data: null_mut(),
        };
        let mut end_image = sd_image_t {
            width: 0,
            height: 0,
            channel: 3,
            data: null_mut(),
        };

        let mut layers = config.skip_layer.clone();
        let guidance = make_guidance(
            config.cfg_scale,
            config.guidance,
            config.slg_scale,
            &mut layers,
            config.skip_layer_start,
            config.skip_layer_end,
        );
        let mut high_noise_layers = config.high_noise_skip_layer.clone();
        let high_noise_guidance = make_guidance(
            config.high_noise_cfg_scale,
            config.high_noise_guidance,
            config.high_noise_slg_scale,
            &mut high_noise_layers,
            config.high_noise_skip_layer_start,
            config.high_noise_skip_layer_end,
        );

        let sample_method = if config.sampling_method == SampleMethod::SAMPLE_METHOD_COUNT {
            sd_get_default_sample_method(sd_ctx)
        } else {
            config.sampling_method
        };
        let high_noise_sample_method =
            if config.high_noise_sampling_method == SampleMethod::SAMPLE_METHOD_COUNT {
                sd_get_default_sample_method(sd_ctx)
            } else {
                config.high_noise_sampling_method
            };
        let scheduler = if model_config.scheduler == Scheduler::SCHEDULER_COUNT {
            sd_get_default_scheduler(sd_ctx, sample_method)
        } else {
            model_config.scheduler
        };

        let sample_params = sd_sample_params_t {
            guidance,
            sample_method,
            sample_steps: config.steps,
            eta: config.eta,
            scheduler,
            shifted_timestep: model_config.timestep_shift,
            custom_sigmas: model_config.sigmas.as_mut_ptr(),
            custom_sigmas_count: model_config.sigmas.len() as i32,
            flow_shift: model_config.flow_shift,
            extra_sample_args: model_config.extra_sample_params.as_optional_ptr(),
        };
        let high_noise_sample_params = sd_sample_params_t {
            guidance: high_noise_guidance,
            sample_method: high_noise_sample_method,
            sample_steps: config.high_noise_steps,
            eta: config.high_noise_eta,
            scheduler,
            shifted_timestep: model_config.timestep_shift,
            custom_sigmas: null_mut(),
            custom_sigmas_count: 0,
            flow_shift: model_config.flow_shift,
            extra_sample_args: config.high_noise_extra_sample_params.as_optional_ptr(),
        };
        let vae_tiling_params = sd_tiling_params_t {
            enabled: model_config.vae_tiling,
            temporal_tiling: model_config.vae_temporal_tiling,
            tile_size_x: model_config.vae_tile_size.0,
            tile_size_y: model_config.vae_tile_size.1,
            target_overlap: model_config.vae_tile_overlap,
            rel_size_x: model_config.vae_relative_tile_size.0,
            rel_size_y: model_config.vae_relative_tile_size.1,
            extra_tiling_args: model_config.vae_extra_tiling_args.as_optional_ptr(),
        };

        let mut _init_buffer: Option<Vec<u8>> = None;
        let mut _end_buffer: Option<Vec<u8>> = None;
        let mut control_frame_storage: Vec<Vec<u8>> = Vec::new();
        let mut control_frames: Vec<sd_image_t> = Vec::new();

        if has_init_image {
            let img = load_rgb_image(&config.init_img)?;
            let (width, height) = img.dimensions();
            let mut buffer = img.into_raw();
            init_image = sd_image_t {
                width,
                height,
                channel: 3,
                data: buffer.as_mut_ptr(),
            };
            _init_buffer = Some(buffer);
        }

        if has_end_image {
            let img = load_rgb_image(&config.end_img)?;
            let (width, height) = img.dimensions();
            let mut buffer = img.into_raw();
            end_image = sd_image_t {
                width,
                height,
                channel: 3,
                data: buffer.as_mut_ptr(),
            };
            _end_buffer = Some(buffer);
        }

        for control_image in &config.control_frames {
            let img = load_rgb_image(control_image)?;
            let (width, height) = img.dimensions();
            let frame = img.into_raw();
            control_frame_storage.push(frame);
            let storage_ref = control_frame_storage.last_mut().unwrap();
            control_frames.push(sd_image_t {
                width,
                height,
                channel: 3,
                data: storage_ref.as_mut_ptr(),
            });
        }

        if config.preview_mode != PreviewType::PREVIEW_NONE && ext_sender.is_some() {
            unsafe extern "C" fn send_preview_frames(
                _step: ::std::os::raw::c_int,
                frame_count: ::std::os::raw::c_int,
                frames: *mut sd_image_t,
                _is_noisy: bool,
                data: *mut ::std::os::raw::c_void,
            ) {
                let _ = catch_unwind(|| unsafe {
                    let sender = &*data.cast::<Option<Sender<ExtProgress>>>();
                    let Some(sender) = sender else {
                        return;
                    };
                    let images = images_from_sd_frames(frames, frame_count);
                    let _ = sender.send(ExtProgress::PreviewFrames(images));
                });
            }

            sd_set_preview_callback(
                Some(send_preview_frames),
                config.preview_mode,
                config.preview_interval,
                !config.preview_noisy,
                config.preview_noisy,
                &mut ext_sender as *mut _ as *mut c_void,
            );
        }

        if sender.is_some() {
            unsafe extern "C" fn progress_callback(
                step: ::std::os::raw::c_int,
                steps: ::std::os::raw::c_int,
                time: f32,
                data: *mut ::std::os::raw::c_void,
            ) {
                let _ = catch_unwind(|| unsafe {
                    let sender = &*data.cast::<Option<Sender<Progress>>>();

                    if let Some(sender) = sender {
                        let _ = sender.send(Progress { step, steps, time });
                    }
                });
            }
            let sender_ptr: *mut c_void = &mut sender as *mut _ as *mut c_void;
            sd_set_progress_callback(Some(progress_callback), sender_ptr);
        } else if ext_sender.is_some() {
            unsafe extern "C" fn ext_progress_callback(
                step: ::std::os::raw::c_int,
                steps: ::std::os::raw::c_int,
                time: f32,
                data: *mut ::std::os::raw::c_void,
            ) {
                let _ = catch_unwind(|| unsafe {
                    let sender = &*data.cast::<Option<Sender<ExtProgress>>>();

                    if let Some(sender) = sender {
                        let _ = sender.send(ExtProgress::Progress(Progress { step, steps, time }));
                    }
                });
            }
            let sender_ptr: *mut c_void = &mut ext_sender as *mut _ as *mut c_void;
            sd_set_progress_callback(Some(ext_progress_callback), sender_ptr);
        }

        let loras: Vec<sd_lora_t> = model_config
            .lora_models
            .iter()
            .map(|(c_path, spec)| sd_lora_t {
                is_high_noise: spec.is_high_noise,
                multiplier: spec.multiplier,
                path: c_path.as_ptr(),
            })
            .collect();

        let mut cache = config.cache.0;
        if let Some(scm_mask) = &config.cache.1 {
            cache.scm_mask = scm_mask.as_ptr();
        }

        let mut num_frames_out = 0;
        let mut hires_custom_sigmas = config.hires_params.1.custom_sigmas.clone();
        let hires = make_hires_params(
            config.hires_params.0,
            &config.hires_params.1,
            config.hires_params.2.as_ref(),
            &mut hires_custom_sigmas,
        );
        let sd_vid_gen_params = sd_vid_gen_params_t {
            loras: loras.as_ptr(),
            lora_count: loras.len() as u32,
            prompt: prompt.as_ptr(),
            negative_prompt: config.negative_prompt.as_ptr(),
            clip_skip: config.clip_skip as i32,
            init_image,
            end_image,
            control_frames: control_frames.as_mut_ptr(),
            control_frames_size: control_frames.len() as i32,
            width: config.width,
            height: config.height,
            sample_params,
            high_noise_sample_params,
            moe_boundary: config.moe_boundary,
            strength: config.strength,
            seed: config.seed,
            video_frames: config.video_frames,
            fps: config.fps,
            vace_strength: config.vace_strength,
            vae_tiling_params,
            cache,
            hires,
        };

        let mut frames_out: *mut sd_image_t = null_mut();
        let mut audio_out: *mut sd_audio_t = null_mut();
        let success = generate_video(
            sd_ctx,
            &sd_vid_gen_params,
            &mut frames_out,
            &mut num_frames_out,
            &mut audio_out,
        );
        let ret = {
            if !success || frames_out.is_null() {
                return Err(DiffusionError::Forward);
            }
            let final_frames = images_from_sd_frames(frames_out, num_frames_out);
            if let Some(sender) = ext_sender.as_ref() {
                let _ = sender.send(final_video_progress(final_frames, audio_out));
            }
            Ok(())
        };
        if !audio_out.is_null() {
            free_sd_audio(audio_out);
        }
        free(frames_out as *mut c_void);
        ret
    }
}

fn save_img(img: sd_image_t, path: &Path, params: Option<&str>) -> Result<(), DiffusionError> {
    // Thx @wandbrandon
    let len = (img.width * img.height * img.channel) as usize;
    let buffer = unsafe { slice::from_raw_parts(img.data, len).to_vec() };
    let save_state = ImageBuffer::from_raw(img.width, img.height, buffer).map(|img| {
        RgbImage::from(img)
            .save(path)
            .map_err(DiffusionError::StoreImages)
    });
    if let Some(Err(err)) = save_state {
        return Err(err);
    }
    if let Some(params) = params {
        let mut metadata = Metadata::new();
        metadata.set_tag(ExifTag::ImageDescription(params.to_string()));
        metadata.write_to_file(path)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use diffusion_rs_sys::{sd_audio_t, sd_image_t};
    use image::{DynamicImage, ImageBuffer, Rgb, Rgba};
    use std::fs;
    use std::path::PathBuf;
    use std::ptr::null;

    use crate::{
        api::{ConfigBuilderError, ModelConfigBuilder},
        util::download_file_hf_hub,
    };

    use super::{
        AudioBuffer, ConfigBuilder, ExtProgress, HiresParamsBuilder, Upscaler, VideoConfigBuilder,
        audio_from_sd_audio, final_video_progress, gen_img, images_from_sd_frames,
    };

    #[test]
    fn test_required_args_txt2img() {
        assert!(ConfigBuilder::default().build().is_err());
        assert!(ModelConfigBuilder::default().build().is_err());
        ModelConfigBuilder::default()
            .model(PathBuf::from("./test.ckpt"))
            .build()
            .unwrap();

        ConfigBuilder::default()
            .prompt("a lovely cat driving a sport car")
            .build()
            .unwrap();

        assert!(matches!(
            ConfigBuilder::default()
                .prompt("a lovely cat driving a sport car")
                .batch_count(10)
                .build(),
            Err(ConfigBuilderError::ValidationError(_))
        ));

        ConfigBuilder::default()
            .prompt("a lovely cat driving a sport car")
            .build()
            .unwrap();

        ConfigBuilder::default()
            .prompt("a lovely duck drinking water from a bottle")
            .batch_count(2)
            .output(PathBuf::from("./"))
            .build()
            .unwrap();
    }

    #[test]
    fn test_required_args_video() {
        VideoConfigBuilder::default()
            .prompt("a lovely cat driving a sport car")
            .build()
            .unwrap();

        assert!(matches!(
            VideoConfigBuilder::default()
                .prompt("a lovely cat driving a sport car")
                .video_frames(0)
                .build(),
            Err(ConfigBuilderError::ValidationError(_))
        ));
    }

    #[test]
    fn test_video_control_frame_validation() {
        let control_path = PathBuf::from("video_control_validation_test.png");
        let frame = ImageBuffer::from_pixel(2, 2, Rgb([255u8, 0, 0]));
        frame.save(&control_path).unwrap();

        let result = VideoConfigBuilder::default()
            .prompt("a lovely cat driving a sport car")
            .control_frames(vec![control_path.clone()])
            .build();

        fs::remove_file(&control_path).unwrap();
        assert!(result.is_ok());
    }

    #[test]
    fn test_in_memory_image_sources_build() {
        let image = ImageBuffer::from_pixel(2, 2, Rgb([255u8, 0, 0]));

        let config = ConfigBuilder::default()
            .prompt("a lovely cat driving a sport car")
            .init_img(image.clone())
            .mask_img(image.clone())
            .ref_images(vec![image.clone()])
            .build();
        assert!(config.is_ok());

        let video_config = VideoConfigBuilder::default()
            .prompt("a lovely cat driving a sport car")
            .init_img(image.clone())
            .end_img(image.clone())
            .control_frames(vec![image])
            .build();
        assert!(video_config.is_ok());
    }

    #[test]
    fn test_new_api_params_build() {
        let hires = HiresParamsBuilder::default()
            .custom_sigmas(vec![1.0, 0.5])
            .build()
            .unwrap();

        let model_config = ModelConfigBuilder::default()
            .model(PathBuf::from("./test.ckpt"))
            .audio_vae(PathBuf::from("./audio_vae.gguf"))
            .embeddings_connectors(PathBuf::from("./connectors"))
            .vae_temporal_tiling(true)
            .vae_extra_tiling_args("{\"temporal\":true}")
            .extra_sample_params("{\"sample\":true}")
            .backend("cpu")
            .params_backend("cpu")
            .hires_params(Upscaler::SD_HIRES_UPSCALER_LATENT, hires.clone(), None)
            .build()
            .unwrap();

        assert_eq!(
            PathBuf::from(&model_config.audio_vae),
            PathBuf::from("./audio_vae.gguf")
        );
        assert_eq!(
            PathBuf::from(&model_config.embeddings_connectors),
            PathBuf::from("./connectors")
        );
        assert!(model_config.vae_temporal_tiling);
        assert_eq!(
            model_config.vae_extra_tiling_args.0.to_str().unwrap(),
            "{\"temporal\":true}"
        );
        assert_eq!(
            model_config.extra_sample_params.0.to_str().unwrap(),
            "{\"sample\":true}"
        );
        assert_eq!(model_config.backend.0.to_str().unwrap(), "cpu");
        assert_eq!(model_config.params_backend.0.to_str().unwrap(), "cpu");
        assert_eq!(model_config.hires_params.1.custom_sigmas, vec![1.0, 0.5]);

        let video_config = VideoConfigBuilder::default()
            .prompt("a lovely cat driving a sport car")
            .fps(24)
            .high_noise_extra_sample_params("{\"high_noise\":true}")
            .video_hires_params(Upscaler::SD_HIRES_UPSCALER_LATENT, hires, None)
            .build()
            .unwrap();
        assert_eq!(video_config.fps, 24);
        assert_eq!(
            video_config
                .high_noise_extra_sample_params
                .0
                .to_str()
                .unwrap(),
            "{\"high_noise\":true}"
        );
        assert_eq!(video_config.hires_params.1.custom_sigmas, vec![1.0, 0.5]);

        let rebuilt = VideoConfigBuilder::from(video_config).build().unwrap();
        assert_eq!(rebuilt.fps, 24);
        assert_eq!(rebuilt.hires_params.1.custom_sigmas, vec![1.0, 0.5]);
    }

    #[test]
    fn test_images_from_sd_frames() {
        let mut pixels_a = vec![255, 0, 0, 0, 255, 0];
        let mut pixels_b = vec![0, 0, 255, 255, 255, 0];
        let mut frames = vec![
            sd_image_t {
                width: 2,
                height: 1,
                channel: 3,
                data: pixels_a.as_mut_ptr(),
            },
            sd_image_t {
                width: 2,
                height: 1,
                channel: 3,
                data: pixels_b.as_mut_ptr(),
            },
        ];

        let images = images_from_sd_frames(frames.as_mut_ptr(), frames.len() as i32);
        assert_eq!(images.len(), 2);
        assert_eq!(images[0].get_pixel(0, 0).0, [255, 0, 0]);
        assert_eq!(images[1].get_pixel(1, 0).0, [255, 255, 0]);
    }

    #[test]
    fn test_audio_from_sd_audio() {
        let mut samples = vec![0.25, -0.25, 0.5, -0.5];
        let audio = sd_audio_t {
            sample_rate: 44_100,
            channels: 2,
            sample_count: 2,
            data: samples.as_mut_ptr(),
        };

        let owned = audio_from_sd_audio(&audio).unwrap();
        assert_eq!(
            owned,
            AudioBuffer {
                sample_rate: 44_100,
                channels: 2,
                samples,
            }
        );
    }

    #[test]
    fn test_final_video_progress_without_audio() {
        let frame = ImageBuffer::from_pixel(1, 1, Rgb([10u8, 20, 30]));

        match final_video_progress(vec![frame], null()) {
            ExtProgress::FinalFrames(frames) => {
                assert_eq!(frames.len(), 1);
                assert_eq!(frames[0].get_pixel(0, 0).0, [10, 20, 30]);
            }
            other => panic!("expected FinalFrames, got {other:?}"),
        }
    }

    #[test]
    fn test_final_video_progress_with_audio() {
        let frame_a = ImageBuffer::from_pixel(1, 1, Rgb([1u8, 2, 3]));
        let frame_b = ImageBuffer::from_pixel(1, 1, Rgb([4u8, 5, 6]));
        let mut samples = vec![0.1, 0.2, 0.3, 0.4];
        let audio = sd_audio_t {
            sample_rate: 48_000,
            channels: 2,
            sample_count: 2,
            data: samples.as_mut_ptr(),
        };

        match final_video_progress(vec![frame_a, frame_b], &audio) {
            ExtProgress::FinalFramesWithAudio(frames, audio) => {
                assert_eq!(frames.len(), 2);
                assert_eq!(frames[0].get_pixel(0, 0).0, [1, 2, 3]);
                assert_eq!(frames[1].get_pixel(0, 0).0, [4, 5, 6]);
                assert_eq!(audio.sample_rate, 48_000);
                assert_eq!(audio.channels, 2);
                assert_eq!(audio.samples, samples);
            }
            other => panic!("expected FinalFramesWithAudio, got {other:?}"),
        }
    }

    #[ignore]
    #[test]
    fn test_img2img_gen() {
        let model_path =
            download_file_hf_hub("CompVis/stable-diffusion-v-1-4-original", "sd-v1-4.ckpt")
                .unwrap();
        let gen_img_output = "./output_img.png";
        let config = ConfigBuilder::default()
            .prompt("A high quality 3d texture")
            .output(PathBuf::from(gen_img_output))
            .batch_count(1)
            .build()
            .unwrap();

        let mut model_config = ModelConfigBuilder::default()
            .model(model_path)
            .build()
            .unwrap();

        gen_img(&config, &mut model_config).unwrap();

        // 2. Create conditioning image: Gradient square
        let mut cond = ImageBuffer::new(512, 512);
        for (x, y, pixel) in cond.enumerate_pixels_mut() {
            let r = (x as f32 / 512.0 * 255.0) as u8;
            let g = (y as f32 / 512.0 * 255.0) as u8;
            let b = 127;
            *pixel = Rgba([r, g, b, 255]);
        }
        let cond_path = "test_cond_image.png";
        DynamicImage::ImageRgba8(cond)
            .save(cond_path)
            .expect("Failed to save reference image");

        // 3. Call refinement using the generated image as input
        let refine_prompt = "PBR texture map, matching the lighting and micro-detail density of the reference image.";
        let img2img_config = ConfigBuilder::default()
            .prompt(refine_prompt)
            .output(PathBuf::from("./output_img_ref.png"))
            .ref_images(vec![PathBuf::from(cond_path)])
            .init_img(PathBuf::from(gen_img_output))
            .batch_count(1)
            .build()
            .unwrap();
        gen_img(&img2img_config, &mut model_config).unwrap();

        // 4. Ensure decoder only mode works after img2img generation
        gen_img(&config, &mut model_config).unwrap();
    }

    #[ignore]
    #[test]
    fn test_img_gen() {
        let model_path =
            download_file_hf_hub("CompVis/stable-diffusion-v-1-4-original", "sd-v1-4.ckpt")
                .unwrap();

        let upscaler_path = download_file_hf_hub(
            "ximso/RealESRGAN_x4plus_anime_6B",
            "RealESRGAN_x4plus_anime_6B.pth",
        )
        .unwrap();
        let config = ConfigBuilder::default()
            .prompt("a lovely duck drinking water from a bottle")
            .output(PathBuf::from("./output_1.png"))
            .batch_count(1)
            .build()
            .unwrap();
        let mut model_config = ModelConfigBuilder::default()
            .model(model_path)
            .upscale_model(upscaler_path)
            .upscale_repeats(1)
            .build()
            .unwrap();

        gen_img(&config, &mut model_config).unwrap();
        let config2 = ConfigBuilder::from(&config)
            .prompt("a lovely duck drinking water from a straw")
            .output(PathBuf::from("./output_2.png"))
            .build()
            .unwrap();
        gen_img(&config2, &mut model_config).unwrap();

        let config3 = ConfigBuilder::from(&config)
            .prompt("a lovely dog drinking water from a starbucks cup")
            .batch_count(2)
            .output(PathBuf::from("./"))
            .build()
            .unwrap();

        gen_img(&config3, &mut model_config).unwrap();
    }
}
