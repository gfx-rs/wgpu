use std::{
    fmt,
    hash,
    marker::PhantomData,
};

/// An unique index in the storage array that a token points to.
///
/// This type is independent of `spirv::Word`. `spirv::Word` is used in data
/// representation. It holds a SPIR-V and refers to that instruction. In
/// structured representation, we use Token to refer to an SPIR-V instruction.
/// Index is an implementation detail to Token.
type Index = u32;

/// A strongly typed reference to a SPIR-V element.
pub struct Token<T> {
    index: Index,
    marker: PhantomData<T>,
}

impl<T> Clone for Token<T> {
    fn clone(&self) -> Self {
        Token {
            index: self.index,
            marker: self.marker,
        }
    }
}
impl<T> Copy for Token<T> {}
impl<T> PartialEq for Token<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}
impl<T> Eq for Token<T> {}
impl<T> fmt::Debug for Token<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Token({})", self.index)
    }
}
impl<T> hash::Hash for Token<T> {
    fn hash<H: hash::Hasher>(&self, hasher: &mut H) {
        self.index.hash(hasher)
    }
}

impl<T> Token<T> {
    #[cfg(test)]
    pub const DUMMY: Self = Token {
        index: !0,
        marker: PhantomData,
    };

    pub(crate) fn new(index: Index) -> Self {
        Token {
            index,
            marker: PhantomData,
        }
    }

    pub fn index(&self) -> usize {
        self.index as usize
    }
}

/// A structure holding some kind of SPIR-V entity (e.g., type, constant,
/// instruction, etc.) that can be referenced.
#[derive(Debug)]
pub struct Storage<T> {
    /// Values of this storage.
    data: Vec<T>,
}

impl<T> Storage<T> {
    pub fn new() -> Self {
        Storage {
            data: Vec::new(),
        }
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = (Token<T>, &'a T)> {
        self.data
            .iter()
            .enumerate()
            .map(|(i, v)| (Token::new(i as Index), v))
    }

    /// Adds a new value to the storage, returning a typed token.
    ///
    /// The value is not linked to any SPIR-V module.
    pub fn append(&mut self, value: T) -> Token<T> {
        let index = self.data.len() as Index;
        self.data.push(value);
        Token::new(index)
    }

    /// Adds a value with a check for uniqueness: returns a token pointing to
    /// an existing element if its value matches the given one, or adds a new
    /// element otherwise.
    pub fn fetch_or_append(&mut self, value: T) -> Token<T> where T: PartialEq {
        if let Some(index) = self.data.iter().position(|d| d == &value) {
            Token::new(index as Index)
        } else {
            self.append(value)
        }
    }
}

impl<T> std::ops::Index<Token<T>> for Storage<T> {
    type Output = T;
    fn index(&self, token: Token<T>) -> &T {
        &self.data[token.index as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_non_unique() {
        let mut storage: Storage<f64> = Storage::new();
        let t1 = storage.append(0.0);
        let t2 = storage.append(0.0);
        assert!(t1 != t2);
        assert!(storage[t1] == storage[t2]);
    }

    #[test]
    fn append_unique() {
        let mut storage: Storage<f64> = Storage::new();
        let t1 = storage.append(std::f64::NAN);
        let t2 = storage.append(std::f64::NAN);
        assert!(t1 != t2);
        assert!(storage[t1] != storage[t2]);
    }

    #[test]
    fn fetch_or_append_non_unique() {
        let mut storage: Storage<f64> = Storage::new();
        let t1 = storage.fetch_or_append(0.0);
        let t2 = storage.fetch_or_append(0.0);
        assert!(t1 == t2);
        assert!(storage[t1] == storage[t2])
    }

    #[test]
    fn fetch_or_append_unique() {
        let mut storage: Storage<f64> = Storage::new();
        let t1 = storage.fetch_or_append(std::f64::NAN);
        let t2 = storage.fetch_or_append(std::f64::NAN);
        assert!(t1 != t2);
        assert!(storage[t1] != storage[t2]);
    }
}
