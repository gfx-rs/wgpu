#[derive(Clone, Debug, PartialEq)]
pub enum Coord {
    Normalized,
    Pixel,
}

impl Default for Coord {
    fn default() -> Self {
        Self::Normalized
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Address {
    Repeat,
    MirroredRepeat,
    ClampToEdge,
    ClampToZero,
    ClampToBorder,
}

impl Default for Address {
    fn default() -> Self {
        Self::ClampToEdge
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum BorderColor {
    TransparentBlack,
    OpaqueBlack,
    OpaqueWhite,
}

impl Default for BorderColor {
    fn default() -> Self {
        Self::TransparentBlack
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Filter {
    Nearest,
    Linear,
}

impl Default for Filter {
    fn default() -> Self {
        Self::Nearest
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum CompareFunc {
    Never,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Equal,
    NotEqual,
    Always,
}

impl Default for CompareFunc {
    fn default() -> Self {
        Self::Never
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct InlineSampler {
    pub coord: Coord,
    pub address: [Address; 3],
    pub border_color: BorderColor,
    pub mag_filter: Filter,
    pub min_filter: Filter,
    pub mip_filter: Option<Filter>,
    pub compare_func: CompareFunc,
}
