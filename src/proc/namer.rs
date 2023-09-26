use crate::{arena::Handle, FastHashMap, FastHashSet};
use std::borrow::Cow;
use std::hash::{Hash, Hasher};

pub type EntryPointIndex = u16;
const SEPARATOR: char = '_';

#[derive(Debug, Eq, Hash, PartialEq)]
pub enum NameKey {
    Constant(Handle<crate::Constant>),
    GlobalVariable(Handle<crate::GlobalVariable>),
    Type(Handle<crate::Type>),
    StructMember(Handle<crate::Type>, u32),
    Function(Handle<crate::Function>),
    FunctionArgument(Handle<crate::Function>, u32),
    FunctionLocal(Handle<crate::Function>, Handle<crate::LocalVariable>),
    EntryPoint(EntryPointIndex),
    EntryPointLocal(EntryPointIndex, Handle<crate::LocalVariable>),
    EntryPointArgument(EntryPointIndex, u32),
}

/// This processor assigns names to all the things in a module
/// that may need identifiers in a textual backend.
#[derive(Default)]
pub struct Namer {
    /// The last numeric suffix used for each base name. Zero means "no suffix".
    unique: FastHashMap<String, u32>,
    keywords: FastHashSet<&'static str>,
    keywords_case_insensitive: FastHashSet<AsciiUniCase<&'static str>>,
    reserved_prefixes: Vec<&'static str>,
}

impl Namer {
    /// Return a form of `string` suitable for use as the base of an identifier.
    ///
    /// - Drop leading digits.
    /// - Retain only alphanumeric and `_` characters.
    /// - Avoid prefixes in [`Namer::reserved_prefixes`].
    /// - Replace consecutive `_` characters with a single `_` character.
    ///
    /// The return value is a valid identifier prefix in all of Naga's output languages,
    /// and it never ends with a `SEPARATOR` character.
    /// It is used as a key into the unique table.
    fn sanitize<'s>(&self, string: &'s str) -> Cow<'s, str> {
        let string = string
            .trim_start_matches(|c: char| c.is_numeric())
            .trim_end_matches(SEPARATOR);

        let base = if !string.is_empty()
            && !string.contains("__")
            && string
                .chars()
                .all(|c: char| c.is_ascii_alphanumeric() || c == '_')
        {
            Cow::Borrowed(string)
        } else {
            let mut filtered = string
                .chars()
                .filter(|&c| c.is_ascii_alphanumeric() || c == '_')
                .fold(String::new(), |mut s, c| {
                    if s.ends_with('_') && c == '_' {
                        return s;
                    }
                    s.push(c);
                    s
                });
            let stripped_len = filtered.trim_end_matches(SEPARATOR).len();
            filtered.truncate(stripped_len);
            if filtered.is_empty() {
                filtered.push_str("unnamed");
            }
            Cow::Owned(filtered)
        };

        for prefix in &self.reserved_prefixes {
            if base.starts_with(prefix) {
                return format!("gen_{base}").into();
            }
        }

        base
    }

    /// Return a new identifier based on `label_raw`.
    ///
    /// The result:
    /// - is a valid identifier even if `label_raw` is not
    /// - conflicts with no keywords listed in `Namer::keywords`, and
    /// - is different from any identifier previously constructed by this
    ///   `Namer`.
    ///
    /// Guarantee uniqueness by applying a numeric suffix when necessary. If `label_raw`
    /// itself ends with digits, separate them from the suffix with an underscore.
    pub fn call(&mut self, label_raw: &str) -> String {
        use std::fmt::Write as _; // for write!-ing to Strings

        let base = self.sanitize(label_raw);
        debug_assert!(!base.is_empty() && !base.ends_with(SEPARATOR));

        // This would seem to be a natural place to use `HashMap::entry`. However, `entry`
        // requires an owned key, and we'd like to avoid heap-allocating strings we're
        // just going to throw away. The approach below double-hashes only when we create
        // a new entry, in which case the heap allocation of the owned key was more
        // expensive anyway.
        match self.unique.get_mut(base.as_ref()) {
            Some(count) => {
                *count += 1;
                // Add the suffix. This may fit in base's existing allocation.
                let mut suffixed = base.into_owned();
                write!(suffixed, "{}{}", SEPARATOR, *count).unwrap();
                suffixed
            }
            None => {
                let mut suffixed = base.to_string();
                if base.ends_with(char::is_numeric)
                    || self.keywords.contains(base.as_ref())
                    || self
                        .keywords_case_insensitive
                        .contains(&AsciiUniCase(base.as_ref()))
                {
                    suffixed.push(SEPARATOR);
                }
                debug_assert!(!self.keywords.contains::<str>(&suffixed));
                // `self.unique` wants to own its keys. This allocates only if we haven't
                // already done so earlier.
                self.unique.insert(base.into_owned(), 0);
                suffixed
            }
        }
    }

    pub fn call_or(&mut self, label: &Option<String>, fallback: &str) -> String {
        self.call(match *label {
            Some(ref name) => name,
            None => fallback,
        })
    }

    /// Enter a local namespace for things like structs.
    ///
    /// Struct member names only need to be unique amongst themselves, not
    /// globally. This function temporarily establishes a fresh, empty naming
    /// context for the duration of the call to `body`.
    fn namespace(&mut self, capacity: usize, body: impl FnOnce(&mut Self)) {
        let fresh = FastHashMap::with_capacity_and_hasher(capacity, Default::default());
        let outer = std::mem::replace(&mut self.unique, fresh);
        body(self);
        self.unique = outer;
    }

    pub fn reset(
        &mut self,
        module: &crate::Module,
        reserved_keywords: &[&'static str],
        extra_reserved_keywords: &[&'static str],
        reserved_keywords_case_insensitive: &[&'static str],
        reserved_prefixes: &[&'static str],
        output: &mut FastHashMap<NameKey, String>,
    ) {
        self.reserved_prefixes.clear();
        self.reserved_prefixes.extend(reserved_prefixes.iter());

        self.unique.clear();
        self.keywords.clear();
        self.keywords.extend(reserved_keywords.iter());
        self.keywords.extend(extra_reserved_keywords.iter());

        debug_assert!(reserved_keywords_case_insensitive
            .iter()
            .all(|s| s.is_ascii()));
        self.keywords_case_insensitive.clear();
        self.keywords_case_insensitive.extend(
            reserved_keywords_case_insensitive
                .iter()
                .map(|string| (AsciiUniCase(*string))),
        );

        let mut temp = String::new();

        for (ty_handle, ty) in module.types.iter() {
            let ty_name = self.call_or(&ty.name, "type");
            output.insert(NameKey::Type(ty_handle), ty_name);

            if let crate::TypeInner::Struct { ref members, .. } = ty.inner {
                // struct members have their own namespace, because access is always prefixed
                self.namespace(members.len(), |namer| {
                    for (index, member) in members.iter().enumerate() {
                        let name = namer.call_or(&member.name, "member");
                        output.insert(NameKey::StructMember(ty_handle, index as u32), name);
                    }
                })
            }
        }

        for (ep_index, ep) in module.entry_points.iter().enumerate() {
            let ep_name = self.call(&ep.name);
            output.insert(NameKey::EntryPoint(ep_index as _), ep_name);
            for (index, arg) in ep.function.arguments.iter().enumerate() {
                let name = self.call_or(&arg.name, "param");
                output.insert(
                    NameKey::EntryPointArgument(ep_index as _, index as u32),
                    name,
                );
            }
            for (handle, var) in ep.function.local_variables.iter() {
                let name = self.call_or(&var.name, "local");
                output.insert(NameKey::EntryPointLocal(ep_index as _, handle), name);
            }
        }

        for (fun_handle, fun) in module.functions.iter() {
            let fun_name = self.call_or(&fun.name, "function");
            output.insert(NameKey::Function(fun_handle), fun_name);
            for (index, arg) in fun.arguments.iter().enumerate() {
                let name = self.call_or(&arg.name, "param");
                output.insert(NameKey::FunctionArgument(fun_handle, index as u32), name);
            }
            for (handle, var) in fun.local_variables.iter() {
                let name = self.call_or(&var.name, "local");
                output.insert(NameKey::FunctionLocal(fun_handle, handle), name);
            }
        }

        for (handle, var) in module.global_variables.iter() {
            let name = self.call_or(&var.name, "global");
            output.insert(NameKey::GlobalVariable(handle), name);
        }

        for (handle, constant) in module.constants.iter() {
            let label = match constant.name {
                Some(ref name) => name,
                None => {
                    use std::fmt::Write;
                    // Try to be more descriptive about the constant values
                    temp.clear();
                    write!(temp, "const_{}", output[&NameKey::Type(constant.ty)]).unwrap();
                    &temp
                }
            };
            let name = self.call(label);
            output.insert(NameKey::Constant(handle), name);
        }
    }
}

/// A string wrapper type with an ascii case insensitive Eq and Hash impl
struct AsciiUniCase<S: AsRef<str> + ?Sized>(S);

impl<S: AsRef<str>> PartialEq<Self> for AsciiUniCase<S> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0.as_ref().eq_ignore_ascii_case(other.0.as_ref())
    }
}

impl<S: AsRef<str>> Eq for AsciiUniCase<S> {}

impl<S: AsRef<str>> Hash for AsciiUniCase<S> {
    #[inline]
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        for byte in self
            .0
            .as_ref()
            .as_bytes()
            .iter()
            .map(|b| b.to_ascii_lowercase())
        {
            hasher.write_u8(byte);
        }
    }
}

#[test]
fn test() {
    let mut namer = Namer::default();
    assert_eq!(namer.call("x"), "x");
    assert_eq!(namer.call("x"), "x_1");
    assert_eq!(namer.call("x1"), "x1_");
    assert_eq!(namer.call("__x"), "_x");
    assert_eq!(namer.call("1___x"), "_x_1");
}
