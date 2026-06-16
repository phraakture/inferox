use std::fmt;
use std::io;

/// Errors for parser failure
#[derive(Debug)]
pub enum Error {
    /// An I/O error occurred while reading the file.
    Io(io::Error),

    /// The file does not start with the GGUF magic bytes.
    InvalidMagic([u8; 4]),

    /// The GGUF version is not supported by this parser.
    UnsupportedVersion(u32),

    /// A metadata value type id is unknown.
    UnknownValueType(u32),

    /// A tensor type id is unknown.
    UnknownTensorType(u32),

    /// A string in the file is not valid UTF-8.
    InvalidUtf8(std::string::FromUtf8Error),

    /// The tensor shape claims more dimensions than GGUF allows.
    InvalidShape { n_dims: u32 },

    /// The file ended before the parser could read the requested data.
    UnexpectedEof,

    /// A requested tensor name was not found in the tensor info table.
    TensorNotFound(String),

    /// A requested tensor index is outside the tensor info table.
    InvalidTensorIndex(usize),

    /// A tensor's byte range overflows the available address space or file bounds.
    TensorOffsetOverflow,

    /// The model architecture in the GGUF file is not supported.
    UnsupportedArchitecture(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io error: {e}"),
            Error::InvalidMagic(m) => {
                write!(f, "invalid magic: expected 'GGUF', got {:?}", m)
            }
            Error::UnsupportedVersion(v) => {
                write!(f, "unsupported GGUF version: {v} (expected 2 or 3)")
            }
            Error::UnknownValueType(t) => write!(f, "unknown metadata value type id: {t}"),
            Error::UnknownTensorType(t) => write!(f, "unknown tensor type id: {t}"),
            Error::InvalidUtf8(e) => write!(f, "invalid utf-8 string: {e}"),
            Error::InvalidShape { n_dims } => {
                write!(f, "tensor shape has invalid number of dimensions: {n_dims}")
            }
            Error::UnexpectedEof => write!(f, "unexpected end of file"),
            Error::TensorNotFound(name) => write!(f, "tensor not found: {name}"),
            Error::InvalidTensorIndex(index) => {
                write!(f, "tensor index out of bounds: {index}")
            }
            Error::TensorOffsetOverflow => write!(f, "tensor byte range overflows file bounds"),
            Error::UnsupportedArchitecture(arch) => {
                write!(f, "unsupported model architecture: {arch}")
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            Error::InvalidUtf8(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<std::string::FromUtf8Error> for Error {
    fn from(e: std::string::FromUtf8Error) -> Self {
        Error::InvalidUtf8(e)
    }
}

/// Shorthand for `Result<T, gguf::Error>`.
pub type Result<T> = std::result::Result<T, Error>;
