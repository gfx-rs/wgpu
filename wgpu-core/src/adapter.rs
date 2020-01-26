use crate::{
    backend,
    hub::{GfxBackend, Global, IdentityFilter, Token},
    id::AdapterId,
    Backend,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub use hal::adapter::AdapterInfo;
use hal::Instance;

#[derive(Debug)]
pub struct RawAdapter<B: hal::Backend> {
    pub(crate) raw: hal::adapter::Adapter<B>,
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub(crate) enum Adapter {
    #[cfg(any(
        not(any(target_os = "ios", target_os = "macos")),
        feature = "gfx-backend-vulkan"
    ))]
    Vulkan(RawAdapter<gfx_backend_vulkan::Backend>),
    #[cfg(any(target_os = "ios", target_os = "macos"))]
    Metal(RawAdapter<gfx_backend_metal::Backend>),
    #[cfg(windows)]
    Dx12(RawAdapter<gfx_backend_dx12::Backend>),
    #[cfg(windows)]
    Dx11(RawAdapter<gfx_backend_dx11::Backend>),
}

impl Adapter {
    pub(crate) fn enumerate<F: IdentityFilter<AdapterId>>(
        global: &Global<F>,
        inputs: AdapterInputs<F::Input>,
    ) -> (Vec<Adapter>, BackendIds<F>) {
        let instance = &global.instance;
        let backend_ids = BackendIds::<F>::new(inputs);
        let mut adapters = Vec::new();

        #[cfg(any(
            not(any(target_os = "ios", target_os = "macos")),
            feature = "gfx-backend-vulkan"
        ))]
        {
            if let Some(ref inst) = instance.vulkan {
                if backend_ids.vulkan.is_some() {
                    for raw in inst.enumerate_adapters() {
                        adapters.push(Adapter::Vulkan(RawAdapter { raw }));
                    }
                }
            }
        }
        #[cfg(any(target_os = "ios", target_os = "macos"))]
        {
            if backend_ids.metal.is_some() {
                for raw in instance.metal.enumerate_adapters() {
                    adapters.push(Adapter::Metal(RawAdapter { raw }));
                }
            }
        }
        #[cfg(windows)]
        {
            if let Some(ref inst) = instance.dx12 {
                if backend_ids.dx12.is_some() {
                    for raw in inst.enumerate_adapters() {
                        adapters.push(Adapter::Dx12(RawAdapter { raw }));
                    }
                }
            }

            if backend_ids.dx11.is_some() {
                for raw in instance.dx11.enumerate_adapters() {
                    adapters.push(Adapter::Dx11(RawAdapter { raw }));
                }
            }
        }

        if adapters.is_empty() {
            log::warn!("No adapters are available!");
        }

        (adapters, backend_ids)
    }

    pub(crate) fn info(&self) -> AdapterInfo {
        match self {
            #[cfg(any(
                not(any(target_os = "ios", target_os = "macos")),
                feature = "gfx-backend-vulkan"
            ))]
            Self::Vulkan(adapter) => adapter.raw.info.clone(),
            #[cfg(any(target_os = "ios", target_os = "macos"))]
            Self::Metal(adapter) => adapter.raw.info.clone(),
            #[cfg(windows)]
            Self::Dx12(adapter) => adapter.raw.info.clone(),
            #[cfg(windows)]
            Self::Dx11(adapter) => adapter.raw.info.clone(),
        }
    }

    pub(crate) fn register<F: IdentityFilter<AdapterId>>(
        self,
        global: &Global<F>,
        backend_ids: BackendIds<F>,
    ) -> AdapterId {
        let mut token = Token::root();

        match self {
            #[cfg(any(
                not(any(target_os = "ios", target_os = "macos")),
                feature = "gfx-backend-vulkan"
            ))]
            Adapter::Vulkan(adapter) => {
                log::info!("Adapter Vulkan {:?}", adapter.raw.info);
                backend::Vulkan::hub(global).adapters.register_identity(
                    backend_ids.vulkan.unwrap(),
                    adapter,
                    &mut token,
                )
            }
            #[cfg(any(target_os = "ios", target_os = "macos"))]
            Adapter::Metal(adapter) => {
                log::info!("Adapter Metal {:?}", adapter.raw.info);
                backend::Metal::hub(global).adapters.register_identity(
                    backend_ids.metal.unwrap(),
                    adapter,
                    &mut token,
                )
            }
            #[cfg(windows)]
            Adapter::Dx12(adapter) => {
                log::info!("Adapter Dx12 {:?}", adapter.raw.info);
                backend::Dx12::hub(global).adapters.register_identity(
                    backend_ids.dx12.unwrap(),
                    adapter,
                    &mut token,
                )
            }
            #[cfg(windows)]
            Adapter::Dx11(adapter) => {
                log::info!("Adapter Dx11 {:?}", adapter.raw.info);
                backend::Dx11::hub(global).adapters.register_identity(
                    backend_ids.dx11.unwrap(),
                    adapter,
                    &mut token,
                )
            }
        }
    }
}

pub enum AdapterInputs<'a, I> {
    IdSet(&'a [I], fn(&I) -> Backend),
    Mask(BackendBit, fn() -> I),
}

impl<I: Clone> AdapterInputs<'_, I> {
    fn find(&self, b: Backend) -> Option<I> {
        match *self {
            AdapterInputs::IdSet(ids, ref fun) => ids.iter().find(|id| fun(id) == b).cloned(),
            AdapterInputs::Mask(bits, ref fun) => {
                if bits.contains(b.into()) {
                    Some(fun())
                } else {
                    None
                }
            }
        }
    }
}

bitflags::bitflags! {
    #[repr(transparent)]
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    pub struct BackendBit: u32 {
        const VULKAN = 1 << Backend::Vulkan as u32;
        const GL = 1 << Backend::Gl as u32;
        const METAL = 1 << Backend::Metal as u32;
        const DX12 = 1 << Backend::Dx12 as u32;
        const DX11 = 1 << Backend::Dx11 as u32;
        /// Vulkan + METAL + DX12
        const PRIMARY = Self::VULKAN.bits | Self::METAL.bits | Self::DX12.bits;
        /// OpenGL + DX11
        const SECONDARY = Self::GL.bits | Self::DX11.bits;
    }
}

impl From<Backend> for BackendBit {
    fn from(backend: Backend) -> Self {
        BackendBit::from_bits(1 << backend as u32).unwrap()
    }
}

pub(crate) struct BackendIds<F: IdentityFilter<AdapterId>> {
    #[cfg(any(
        not(any(target_os = "ios", target_os = "macos")),
        feature = "gfx-backend-vulkan"
    ))]
    pub(crate) vulkan: Option<F::Input>,
    #[cfg(any(target_os = "ios", target_os = "macos"))]
    pub(crate) metal: Option<F::Input>,
    #[cfg(windows)]
    pub(crate) dx12: Option<F::Input>,
    #[cfg(windows)]
    pub(crate) dx11: Option<F::Input>,
}

impl<F: IdentityFilter<AdapterId>> BackendIds<F> {
    pub(crate) fn new(inputs: AdapterInputs<F::Input>) -> Self {
        Self {
            #[cfg(any(
                not(any(target_os = "ios", target_os = "macos")),
                feature = "gfx-backend-vulkan"
            ))]
            vulkan: inputs.find(Backend::Vulkan),
            #[cfg(any(target_os = "ios", target_os = "macos"))]
            metal: inputs.find(Backend::Metal),
            #[cfg(windows)]
            dx12: inputs.find(Backend::Dx12),
            #[cfg(windows)]
            dx11: inputs.find(Backend::Dx11),
        }
    }
}

impl<F: IdentityFilter<AdapterId>> Clone for BackendIds<F> {
    fn clone(&self) -> Self {
        Self {
            #[cfg(any(
                not(any(target_os = "ios", target_os = "macos")),
                feature = "gfx-backend-vulkan"
            ))]
            vulkan: self.vulkan.clone(),
            #[cfg(any(target_os = "ios", target_os = "macos"))]
            metal: self.metal.clone(),
            #[cfg(windows)]
            dx12: self.dx12.clone(),
            #[cfg(windows)]
            dx11: self.dx11.clone(),
        }
    }
}
