/*!
Frontend parsers that consume binary and text shaders and load them into [`Module`](super::Module)s.
*/

mod interpolator;

#[cfg(feature = "glsl-in")]
pub mod glsl;
#[cfg(feature = "spv-in")]
pub mod spv;
#[cfg(feature = "wgsl-in")]
pub mod wgsl;

use crate::{
    arena::{Arena, Handle, UniqueArena},
    proc::{ResolveContext, ResolveError, TypeResolution},
    FastHashMap,
};
use std::ops;

/// Helper class to emit expressions
#[allow(dead_code)]
#[derive(Default, Debug)]
struct Emitter {
    start_len: Option<usize>,
}

#[allow(dead_code)]
impl Emitter {
    fn start(&mut self, arena: &Arena<crate::Expression>) {
        if self.start_len.is_some() {
            unreachable!("Emitting has already started!");
        }
        self.start_len = Some(arena.len());
    }
    #[must_use]
    fn finish(
        &mut self,
        arena: &Arena<crate::Expression>,
    ) -> Option<(crate::Statement, crate::span::Span)> {
        let start_len = self.start_len.take().unwrap();
        if start_len != arena.len() {
            #[allow(unused_mut)]
            let mut span = crate::span::Span::default();
            let range = arena.range_from(start_len);
            #[cfg(feature = "span")]
            for handle in range.clone() {
                span.subsume(arena.get_span(handle))
            }
            Some((crate::Statement::Emit(range), span))
        } else {
            None
        }
    }
}

#[allow(dead_code)]
impl super::ConstantInner {
    const fn boolean(value: bool) -> Self {
        Self::Scalar {
            width: super::BOOL_WIDTH,
            value: super::ScalarValue::Bool(value),
        }
    }
}

/// Helper processor that derives the types of all expressions.
#[derive(Debug, Default)]
pub struct Typifier {
    resolutions: Vec<TypeResolution>,
}

impl Typifier {
    pub const fn new() -> Self {
        Typifier {
            resolutions: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.resolutions.clear()
    }

    pub fn get<'a>(
        &'a self,
        expr_handle: Handle<crate::Expression>,
        types: &'a UniqueArena<crate::Type>,
    ) -> &'a crate::TypeInner {
        self.resolutions[expr_handle.index()].inner_with(types)
    }

    pub fn grow(
        &mut self,
        expr_handle: Handle<crate::Expression>,
        expressions: &Arena<crate::Expression>,
        ctx: &ResolveContext,
    ) -> Result<(), ResolveError> {
        if self.resolutions.len() <= expr_handle.index() {
            for (eh, expr) in expressions.iter().skip(self.resolutions.len()) {
                //Note: the closure can't `Err` by construction
                let resolution = ctx.resolve(expr, |h| Ok(&self.resolutions[h.index()]))?;
                log::debug!("Resolving {:?} = {:?} : {:?}", eh, expr, resolution);
                self.resolutions.push(resolution);
            }
        }
        Ok(())
    }

    /// Invalidates the cached type resolution for `expr_handle` forcing a recomputation
    ///
    /// If the type of the expression hasn't yet been calculated a
    /// [`grow`](Self::grow) is performed instead
    pub fn invalidate(
        &mut self,
        expr_handle: Handle<crate::Expression>,
        expressions: &Arena<crate::Expression>,
        ctx: &ResolveContext,
    ) -> Result<(), ResolveError> {
        if self.resolutions.len() <= expr_handle.index() {
            self.grow(expr_handle, expressions, ctx)
        } else {
            let expr = &expressions[expr_handle];
            //Note: the closure can't `Err` by construction
            let resolution = ctx.resolve(expr, |h| Ok(&self.resolutions[h.index()]))?;
            self.resolutions[expr_handle.index()] = resolution;
            Ok(())
        }
    }
}

impl ops::Index<Handle<crate::Expression>> for Typifier {
    type Output = TypeResolution;
    fn index(&self, handle: Handle<crate::Expression>) -> &Self::Output {
        &self.resolutions[handle.index()]
    }
}

/// Type representing a lexical scope, associating a name to a single variable
///
/// The scope is generic over the variable representation and name representaion
/// in order to allow larger flexibility on the frontends on how they might
/// represent them.
type Scope<Name, Var> = FastHashMap<Name, Var>;

/// Structure responsible for managing variable lookups and keeping track of
/// lexical scopes
///
/// The symbol table is generic over the variable representation and its name
/// to allow larger flexibility on the frontends on how they might represent them.
///
/// ```
/// use naga::front::SymbolTable;
///
/// // Create a new symbol table with `u32`s representing the variable
/// let mut symbol_table: SymbolTable<&str, u32> = SymbolTable::default();
///
/// // Add two variables named `var1` and `var2` with 0 and 2 respectively
/// symbol_table.add("var1", 0);
/// symbol_table.add("var2", 2);
///
/// // Check that `var1` exists and is `0`
/// assert_eq!(symbol_table.lookup("var1"), Some(&0));
///
/// // Push a new scope and add a variable to it named `var1` shadowing the
/// // variable of our previous scope
/// symbol_table.push_scope();
/// symbol_table.add("var1", 1);
///
/// // Check that `var1` now points to the new value of `1` and `var2` still
/// // exists with its value of `2`
/// assert_eq!(symbol_table.lookup("var1"), Some(&1));
/// assert_eq!(symbol_table.lookup("var2"), Some(&2));
///
/// // Pop the scope
/// symbol_table.pop_scope();
///
/// // Check that `var1` now refers to our initial variable with value `0`
/// assert_eq!(symbol_table.lookup("var1"), Some(&0));
/// ```
///
/// Scopes are ordered as a LIFO stack so a variable defined in a later scope
/// with the same name as another variable defined in a earlier scope will take
/// precedence in the lookup. Scopes can be added with [`push_scope`] and
/// removed with [`pop_scope`].
///
/// A root scope is added when the symbol table is created and must always be
/// present. Trying to pop it will result in a panic.
///
/// Variables can be added with [`add`] and looked up with [`lookup`]. Adding a
/// variable will do so in the currently active scope and as mentioned
/// previously a lookup will search from the current scope to the root scope.
///
/// [`push_scope`]: Self::push_scope
/// [`pop_scope`]: Self::push_scope
/// [`add`]: Self::add
/// [`lookup`]: Self::lookup
pub struct SymbolTable<Name, Var> {
    /// Stack of lexical scopes. Not all scopes are active; see [`cursor`].
    ///
    /// [`cursor`]: Self::cursor
    scopes: Vec<Scope<Name, Var>>,
    /// Limit of the [`scopes`] stack (exclusive). By using a separate value for
    /// the stack length instead of `Vec`'s own internal length, the scopes can
    /// be reused to cache memory allocations.
    ///
    /// [`scopes`]: Self::scopes
    cursor: usize,
}

impl<Name, Var> SymbolTable<Name, Var> {
    /// Adds a new lexical scope.
    ///
    /// All variables declared after this point will be added to this scope
    /// until another scope is pushed or [`pop_scope`] is called, causing this
    /// scope to be removed along with all variables added to it.
    ///
    /// [`pop_scope`]: Self::pop_scope
    pub fn push_scope(&mut self) {
        // If the cursor is equal to the scope's stack length then we need to
        // push another empty scope. Otherwise we can reuse the already existing
        // scope.
        if self.scopes.len() == self.cursor {
            self.scopes.push(FastHashMap::default())
        } else {
            self.scopes[self.cursor].clear();
        }

        self.cursor += 1;
    }

    /// Removes the current lexical scope and all its variables
    ///
    /// # PANICS
    /// - If the current lexical scope is the root scope
    pub fn pop_scope(&mut self) {
        // Despite the method title, the variables are only deleted when the
        // scope is reused. This is because while a clear is inevitable if the
        // scope needs to be reused, there are cases where the scope might be
        // popped and not reused, i.e. if another scope with the same nesting
        // level is never pushed again.
        assert!(self.cursor != 1, "Tried to pop the root scope");

        self.cursor -= 1;
    }
}

impl<Name, Var> SymbolTable<Name, Var>
where
    Name: std::hash::Hash + Eq,
{
    /// Perform a lookup for a variable named `name`.
    ///
    /// As stated in the struct level documentation the lookup will proceed from
    /// the current scope to the root scope, returning `Some` when a variable is
    /// found or `None` if there doesn't exist a variable with `name` in any
    /// scope.
    pub fn lookup<Q: ?Sized>(&self, name: &Q) -> Option<&Var>
    where
        Name: std::borrow::Borrow<Q>,
        Q: std::hash::Hash + Eq,
    {
        // Iterate backwards trough the scopes and try to find the variable
        for scope in self.scopes[..self.cursor].iter().rev() {
            if let Some(var) = scope.get(name) {
                return Some(var);
            }
        }

        None
    }

    /// Adds a new variable to the current scope.
    ///
    /// Returns the previous variable with the same name in this scope if it
    /// exists, so that the frontend might handle it in case variable shadowing
    /// is disallowed.
    pub fn add(&mut self, name: Name, var: Var) -> Option<Var> {
        self.scopes[self.cursor - 1].insert(name, var)
    }

    /// Adds a new variable to the root scope.
    ///
    /// This is used in GLSL for builtins which aren't known in advance and only
    /// when used for the first time, so there must be a way to add those
    /// declarations to the root unconditionally from the current scope.
    ///
    /// Returns the previous variable with the same name in the root scope if it
    /// exists, so that the frontend might handle it in case variable shadowing
    /// is disallowed.
    pub fn add_root(&mut self, name: Name, var: Var) -> Option<Var> {
        self.scopes[0].insert(name, var)
    }
}

impl<Name, Var> Default for SymbolTable<Name, Var> {
    /// Constructs a new symbol table with a root scope
    fn default() -> Self {
        Self {
            scopes: vec![FastHashMap::default()],
            cursor: 1,
        }
    }
}

use std::fmt;

impl<Name: fmt::Debug, Var: fmt::Debug> fmt::Debug for SymbolTable<Name, Var> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SymbolTable ")?;
        f.debug_list()
            .entries(self.scopes[..self.cursor].iter())
            .finish()
    }
}
